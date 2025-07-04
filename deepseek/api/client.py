# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/10
# @Author      : liuboyuan

import asyncio

from llm_client import LLMClient


async def main():
    """
    主函数，负责初始化客户端并启动聊天循环
    """
    client = LLMClient()
    try:
        # 连接到服务器

        # 启动聊天循环
        print('chat loop started')
        await client.chat_loop()
    finally:
        # 清理资源
        await client.cleanup()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())