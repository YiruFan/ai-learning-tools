import sys
print(f"Python version: {sys.version}")

import google.generativeai as genai

# 设置 API 密钥
GEMINI_API_KEY = "AIzaSyBtioiqo9y7wV43XD0_bL4YO55dMqwiPD0"

try:
    # 配置 API
    genai.configure(api_key=GEMINI_API_KEY)
    print("API configured successfully")
    
    # 创建模型
    model = genai.GenerativeModel('gemini-pro')
    print("Model created successfully")
    
    # 测试生成
    response = model.generate_content("Hello, please respond with a simple 'Hi'")
    print(f"Full response object: {response}")
    print(f"Response text: {response.text}")
    
except Exception as e:
    print(f"详细错误信息: {str(e)}")
    import traceback
    print(traceback.format_exc()) 