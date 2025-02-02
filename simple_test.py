try:
    import google.generativeai as genai
    print("Successfully imported google.generativeai")
    print(f"Version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
except Exception as e:
    print(f"Error: {str(e)}") 