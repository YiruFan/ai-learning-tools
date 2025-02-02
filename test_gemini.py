import os
import google.generativeai as genai

def test_gemini():
    try:
        # 1. 打印版本
        print(f"Package version: {genai.__version__}")
        
        # 2. 配置 API
        genai.configure(api_key="AIzaSyBtioiqo9y7wV43XD0_bL4YO55dMqwiPD0")
        
        # 3. 列出可用模型
        print("\nAvailable models:")
        for m in genai.list_models():
            print(m.name)
        
        # 4. 尝试创建模型
        model = genai.GenerativeModel('gemini-pro')
        print("\nModel created successfully")
        
        # 5. 测试生成
        response = model.generate_content('Say hello!')
        print(f"\nResponse: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_gemini() 