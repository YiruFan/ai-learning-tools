import cv2
import numpy as np
import google.generativeai as genai
from google.cloud import vision, speech
# from pytube import YouTube
# from pydub import AudioSegment
from google.cloud import videointelligence, speech_v1p1beta1
from werkzeug.utils import secure_filename
import os
import json
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import time


# Initialize Flask App
app = Flask(__name__, 
    template_folder='templates',  # Explicitly specify template folder
    static_folder='static'        # Explicitly specify static folder
)

# Add debug information
print("Starting application...")

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "intrepid-tape-449619-h9-e42fe89fff7c.json"
vision_client = vision.ImageAnnotatorClient()

# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyBtioiqo9y7wV43XD0_bL4YO55dMqwiPD0"

try:
    # Configure API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize model
    model = genai.GenerativeModel('gemini-pro')
    
    # Test model with a more specific prompt
    response = model.generate_content("""
        Please respond with a simple "Hello, connection successful!" if you can read this message.
        Keep your response under 10 words.
    """)
    print("Test response:", response.text)
    print("Model initialized successfully!")
    
except Exception as e:
    print(f"Model initialization error: {str(e)}")
    import traceback
    print(traceback.format_exc())
    model = None

# List available models
# for m in genai.list_models():
#     print(m.name)

# Modify OCR configuration
# Windows users may need to set this path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define Upload Folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# Add new configurations and constants
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload file size to 16MB
ALLOWED_VIDEO_DOMAINS = {'youtube.com', 'youtu.be'}

# Set security configurations
app.config['SECRET_KEY'] = os.urandom(24)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Add static folder
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Modify error handling decorators
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File size exceeds limit (16MB max)"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Modify file validation function
def validate_file(file):
    if not file:
        return False, "No file selected"
    if not allowed_file(file.filename):
        return False, "Unsupported file type"
    return True, ""

# Enhance video URL validation
def validate_video_url(url):
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(d in domain for d in ALLOWED_VIDEO_DOMAINS)
    except:
        return False

# Function to extract text using Google Cloud Vision API
def extract_text_from_file(file_path):
    try:
        print(f"Processing file: {file_path}")
        file_ext = file_path.rsplit('.', 1)[1].lower()
        print(f"File type: {file_ext}")
        
        if file_ext == 'pdf':
            print("Processing PDF file...")
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text if text.strip() else "No text detected in PDF"
            except Exception as pdf_error:
                print(f"PDF processing error: {str(pdf_error)}")
                return f"PDF processing failed: {str(pdf_error)}"
            
        else:
            print("Processing image file...")
            image = cv2.imread(file_path)
            if image is None:
                print(f"Cannot read image: {file_path}")
                return "Cannot read image"
            
            print("Converting to grayscale...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Starting OCR...")
            text = pytesseract.image_to_string(gray, lang='eng')
            print(f"OCR result length: {len(text)}")
            return text if text else "No text detected"
            
    except Exception as e:
        print(f"File processing error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"File processing failed: {str(e)}"


# Function to generate AI-powered summaries and mind maps

def generate_next_topics(extracted_text):
    try:
        prompt = f"""
        Based on the following learning content, suggest 3 related topics for further study.
        Format your response as JSON with the following structure:
        {{
            "topics": [
                {{
                    "name": "topic name",
                    "reason": "detailed reason for recommendation",
                    "link": "suggested learning resource URL"
                }},
                ...
            ]
        }}

        Content: {extracted_text}
        """
        
        response = model.generate_content(prompt)
        try:
            # 尝试解析 AI 返回的 JSON
            import json
            result = json.loads(response.text)
            return result
        except:
            # 如果解析失败，返回默认推荐
            return {
                "topics": [
                    {
                        "name": "Advanced Machine Learning",
                        "reason": "Build upon current concepts with advanced ML techniques",
                        "link": "https://www.coursera.org/learn/machine-learning"
                    },
                    {
                        "name": "Deep Learning Specialization",
                        "reason": "Explore neural networks and deep learning applications",
                        "link": "https://www.deeplearning.ai"
                    },
                    {
                        "name": "Applied AI Projects",
                        "reason": "Practice with real-world AI applications",
                        "link": "https://www.kaggle.com/competitions"
                    }
                ]
            }
    except Exception as e:
        print(f"AI recommendation error: {str(e)}")
        return {"error": "Failed to generate recommendations"}

@app.route("/next_link", methods=["POST"])
def get_next_topics():
    try:
        data = request.get_json()
        extracted_text = data.get("extracted_text", "").strip()

        if not extracted_text:
            return jsonify({"error": "No content provided"}), 400

        recommendations = generate_next_topics(extracted_text)
        return jsonify(recommendations)

    except Exception as e:
        print(f"Topic generation error: {str(e)}")
        return jsonify({"error": "Failed to generate topic suggestions"}), 500




@app.route("/summarize_video", methods=["POST"])
def summarize_video():
    try:
        data = request.get_json()
        video_url = data.get("video_url")

        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        # 使用 Gemini 生成视频总结
        prompt = f"""
        Analyze this video and provide:
        1. Main topics covered
        2. A comprehensive summary
        
        Video URL: {video_url}
        
        Format the response as:
        Topics: [List 3-4 main topics]
        Summary: [A detailed summary of the content]
        """
        
        response = model.generate_content(prompt)
        
        # 解析响应
        summary_text = response.text if response and response.text else "Could not generate summary"
        topics = ["Topic 1", "Topic 2", "Topic 3"]  # 可以从响应中提取
        
        return jsonify({
            "success": True,
            "topics": topics,
            "ai_summary": summary_text
        })

    except Exception as e:
        print(f"Video processing error: {str(e)}")
        return jsonify({"error": "Error processing video"}), 500





# Homepage Route
@app.route("/")
def index():
    try:
        current_dir = os.getcwd()
        template_dir = os.path.join(current_dir, 'templates')
        print(f"Current directory: {current_dir}")
        print(f"Template directory: {template_dir}")
        print(f"Template exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
        
        return render_template("index.html")
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        print(f"Template folders: {app.jinja_loader.list_templates()}")
        return f"Error: {str(e)}", 500

# 添加一个测试路由
@app.route("/test")
def test():
    return "Server is running!"

# 添加错误处理路由
@app.errorhandler(403)
def forbidden(e):
    return "Access forbidden. Please check file permissions.", 403

# handling image upload
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from file
            extracted_text = extract_text_from_file(filepath)
            print(f"Extracted text length: {len(extracted_text)}")
            
            # Generate summary
            summary = generate_ai_summary(extracted_text)
            print(f"Generated summary: {summary[:100]}...")  # 打印生成的摘要前100个字符
            
            # Generate mind map
            mind_map = generate_mind_map(extracted_text)
            
            # Generate suggestions
            suggestions = generate_next_topics(extracted_text)

            # Clean up temporary file
            os.remove(filepath)

            return jsonify({
                "success": True,
                "summary": summary,
                "mind_map": mind_map,
                "suggestions": suggestions
            })

        return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

def generate_mind_map(extracted_text):
    try:
        print(f"Generating mind map for text length: {len(extracted_text)}")
        prompt = f"""
        Create a mind map based on the following content. 
        The mind map should have a central topic that summarizes the main theme,
        and 3-4 main branches that represent key aspects or concepts.
        Each branch should have 2-3 specific points or examples.

        Format your response strictly as JSON with this structure:
        {{
            "central_topic": "Main Theme",
            "branches": [
                {{
                    "topic": "Key Aspect 1",
                    "points": ["Specific point 1", "Specific point 2", "Example 1"]
                }},
                {{
                    "topic": "Key Aspect 2",
                    "points": ["Specific point 1", "Specific point 2", "Example 2"]
                }}
            ]
        }}

        Content to analyze: {extracted_text}
        """
        
        print("Sending prompt to AI...")
        response = model.generate_content(prompt)
        print(f"AI Response received: {response.text}")
        
        try:
            mind_map_data = json.loads(response.text)
            print("Successfully parsed JSON response")
            return mind_map_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response.text}")
            # 尝试提取JSON部分
            try:
                # 查找JSON开始和结束的位置
                start = response.text.find('{')
                end = response.text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response.text[start:end]
                    return json.loads(json_str)
            except:
                pass
            
            return {
                "central_topic": "Content Structure",
                "branches": [
                    {
                        "topic": "Main Points",
                        "points": ["Content detected", "Processing in progress", "Try uploading again"]
                    },
                    {
                        "topic": "Suggestions",
                        "points": ["Check file format", "Ensure text is clear", "Try different content"]
                    }
                ]
            }
    except Exception as e:
        print(f"Mind map generation error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "central_topic": "Analysis in Progress",
            "branches": [
                {
                    "topic": "Status",
                    "points": ["Processing content", "Please wait", "Try again if needed"]
                }
            ]
        }

# Modify retry content generation function
def retry_generate_content(prompt, max_retries=3):
    global model
    
    for attempt in range(max_retries):
        try:
            if model is None:
                print(f"Attempt {attempt + 1}: Reinitializing model...")
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-pro')
            
            # Add generation configuration
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                )
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)
            model = None
    
    return "Could not generate content, please try again later"

# Use retry mechanism in summary generation function
def generate_ai_summary(extracted_text):
    if not extracted_text:
        return "Error: No text to analyze"
    
    # Limit input text length
    max_length = 4000
    if len(extracted_text) > max_length:
        extracted_text = extracted_text[:max_length] + "..."
    
    prompt = f"""
    Given the following text, extract the key points and provide a structured summary:

    Text:
    {extracted_text}

    Please provide:
    1. A brief summary (200 words max)
    2. 3-5 key points
    3. Main conclusions
    """
    
    return retry_generate_content(prompt)

# Modify test API route
@app.route("/test_api")
def test_api():
    try:
        if model is None:
            return jsonify({"error": "Model not initialized"}), 500
            
        test_prompt = """
        Please generate a simple test response with:
        1. A greeting
        2. A simple fact
        3. A closing statement
        """
        
        response = model.generate_content(test_prompt)
        return jsonify({
            "success": True,
            "response": response.text
        })
    except Exception as e:
        return jsonify({
            "error": f"API test failed: {str(e)}"
        }), 500

# Run Flask App
if __name__ == "__main__":
    print("Starting Flask server...")
    
    # Disable Flask auto-reloader
    app.config['DEBUG'] = True
    app.config['USE_RELOADER'] = False
    
    # Try different ports
    for port in range(5008, 5020):
        try:
            print(f"Trying port {port}...")
            app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False)
            break
        except OSError as e:
            print(f"Port {port} is in use, trying next port...")
            continue
