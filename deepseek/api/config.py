# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/10
# @Author      : liuboyuan
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
EMBEDDING_MODEL_NAME = 'text-embedding-004'
SEARCH_ENHANCE_MODEL_NAME = 'gemini-2.5-pro-preview-06-05'
DEEPSEEK_MODEL_NAME = 'DeepSeek-R1-0528'

# API配置
gemini_base_url = os.getenv('gemini_base_url')
gemini_api_key = os.getenv('gemini_api_key')
ds_base_url = os.getenv('ds_base_url')
ds_api_key = os.getenv('ds_api_key') 