# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/10
# @Author      : liuboyuan
import os
from contextlib import AsyncExitStack
from typing import List, Tuple, Dict, Any
import json
from openai import OpenAI

# 导入公共配置
from config import (
    EMBEDDING_MODEL_NAME,
    SEARCH_ENHANCE_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    gemini_base_url,
    gemini_api_key,
    ds_base_url,
    ds_api_key
)

# 导入RAG模块
from rag_for_materials import search

class LLMClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_client = OpenAI(
            api_key=ds_api_key,
            base_url=ds_base_url
        )

        self.gemini_client = OpenAI(
            api_key=gemini_api_key,
            base_url=gemini_base_url
        )


        
    def search_database(self, question: str):
        """
        调用rag_for_materials中的search函数搜索数据库
        
        Args:
            question: 用户查询
            
        Returns:
            搜索结果列表
        """
        return search(question)

    async def process_query(self, query: str) -> str:
        """
        处理用户查询请求

        Args:
            query: 用户输入的查询字符串

        Returns:
            str: 处理结果
        """

            
        # 定义搜索工具
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "搜索数据库获取与问题相关的信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "用户的问题或查询"
                        }
                    },
                    "required": ["question"]
                }
            }
        }
        
        # 步骤1：使用模型决定是否调用搜索工具
        try:
            openai_response = self.openai_client.chat.completions.create(
                model="QwQ-32B",
                messages=[
                    {"role": "system", "content": "你是一个智能助手，可以使用工具来回答用户问题。当需要查找信息时，应该使用search_database工具。"},
                    {"role": "user", "content": query}
                ],
                tools=[search_tool],
                tool_choice="auto",
                temperature=0.3
            )
            
            # 检查是否成功调用
            if not openai_response or not openai_response.choices:
                return "模型调用失败，请稍后再试。"

            openai_res_message = openai_response.choices[0].message
            print(openai_res_message)
            # 检查模型是否选择使用工具
            if not openai_res_message.tool_calls:
                return "未调用搜索工具，无法提供准确回答。请尝试重新提问或修改问题。"
            
            # 处理工具调用
            search_results = []
            for tool_call in openai_res_message.tool_calls:
                if tool_call.function.name == "search_database":
                    # 解析工具调用参数
                    try:
                        args = json.loads(tool_call.function.arguments)
                        search_question = args.get("question", query)
                        
                        # 调用实际的搜索函数
                        search_results = self.search_database(search_question)
                    except Exception as e:
                        return f"搜索工具调用失败：{str(e)}"
            
            if not search_results:
                return "未找到相关信息，请尝试其他问题。"
            
            # 步骤2：构建上下文
            print("开始构建上下文")
            context = "\n\n".join([text for text, _ in search_results])

            print("初步分析")
            # 步骤3：将搜索结果返回给DeepSeek进行分析
            llm_analysis_result = self.openai_client.chat.completions.create(
                model=DEEPSEEK_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个强大的助手，能够理解用户的问题并从给定的上下文中提取相关信息。"},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "我需要搜索一些信息来回答你的问题。"},
                    {"role": "assistant", "tool_calls": openai_res_message.tool_calls},
                    {"role": "tool", "tool_call_id": openai_res_message.tool_calls[0].id, "content": json.dumps(search_results, ensure_ascii=False)},
                    {"role": "user", "content": f"基于搜索结果，请分析并回答我的问题：{query}"}
                ],
                temperature=0.3
            )
            
            llm_analysis = llm_analysis_result.choices[0].message.content
            print("总结")
            # 步骤4：使用DeepSeek-R1-0528总结并给出最终答案
            final_response = self.openai_client.chat.completions.create(
                model=DEEPSEEK_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的助手，能够根据提供的信息给出准确、简洁的回答。"},
                    {"role": "user", "content": f"我需要你总结以下内容并给出最终答案。\n\n问题：{query}\n\n搜索结果：\n{context}\n\nQwQ-32B分析：{llm_analysis}\n\n请给出清晰、准确的回答。"}
                ],
                temperature=0.2
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"处理查询时出错：{str(e)}"


    async def chat_loop(self):
        """
        聊天循环，持续接收用户输入并处理
        """
        
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """
        清理资源
        """
        await self.exit_stack.aclose()