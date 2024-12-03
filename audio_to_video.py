import requests
import os
from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, VideoFileClip
from moviepy.video import fx as vfx
import json
from typing import Optional
from config import SILICON_FLOW_API_TOKEN
import argparse
import logging
import colorlog
from tqdm import tqdm

# 免费模型
MODEL_IMAGE = "stabilityai/stable-diffusion-3-5-large"
MODEL_AUDIO = "FunAudioLLM/SenseVoiceSmall"
MODEL_TEXT = "THUDM/glm-4-9b-chat"

# Configure colorful logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Add a file handler for persistent logging
file_handler = logging.FileHandler('audio_to_video.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)


class AudioToVideo:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.siliconflow.cn/v1"
        # Mask API token in logs
        masked_token = f"...{api_token[-4:]}" if api_token else "None"
        logger.info(f"AudioToVideo initialized with API token ending in {masked_token}")

    def _get_masked_headers(self):
        """Return headers with masked API token for logging"""
        masked_headers = self.headers.copy()
        if 'Authorization' in masked_headers:
            token = masked_headers['Authorization'].split()[-1]
            masked_headers['Authorization'] = f"Bearer ...{token[-4:]}"
        return masked_headers

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Convert audio to text using SiliconFlow API"""
        url = f"{self.base_url}/audio/transcriptions"
        logger.info(f"🎤 Starting audio transcription for file: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None
        
        try:
            # Get file size for progress bar
            file_size = os.path.getsize(audio_path)
            
            # Prepare multipart form data with progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="📤 Uploading audio") as pbar:
                class ProgressFileWrapper:
                    def __init__(self, fd):
                        self.fd = fd
                        self.progress = 0

                    def read(self, size=-1):
                        data = self.fd.read(size)
                        if data:
                            pbar.update(len(data))
                        return data

                    def seek(self, *args):
                        return self.fd.seek(*args)

                    def tell(self):
                        return self.fd.tell()

                    def close(self):
                        return self.fd.close()

                with open(audio_path, 'rb') as f:
                    files = {
                        'file': ('audio.mp3', ProgressFileWrapper(f), 'audio/mpeg'),
                        'model': (None, MODEL_AUDIO)
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_token}"
                    }
                    
                    try:
                        logger.debug(f"🌐 Making POST request to {url}")
                        response = requests.post(url, files=files, headers=headers)
                        response.raise_for_status()
                        result = response.json()
                        logger.info("✅ Audio transcription successful")
                        logger.debug(f"📝 Transcription result: {result}")
                        return result.get('text')
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ Network error during transcription: {str(e)}", exc_info=True)
                        return None
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Error parsing API response: {str(e)}", exc_info=True)
                        return None
                
        except Exception as e:
            logger.error(f"❌ Unexpected error in transcription: {str(e)}", exc_info=True)
            return None

    def optimize_text(self, text: str) -> Optional[str]:
        """Optimize text for image generation using SiliconFlow API"""
        url = f"{self.base_url}/chat/completions"
        logger.info("Starting text optimization")
        logger.debug(f"Input text: {text}")
        
        prompt = f"""请基于以下文本，生成一幅画面的详细描述。描述要具体且富有视觉细节：

{text}

请用英文回答，因为后续需要给text-to-image模型使用。字数控制在100字以内"""
        
        payload = {
            "model": MODEL_TEXT,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }

        try:
            logger.debug(f"Making POST request to {url} with payload: {json.dumps(payload, indent=2)}")
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            optimized_text = result.get('choices', [{}])[0].get('message', {}).get('content')
            logger.info("Text optimization successful")
            logger.debug(f"Optimized text: {optimized_text}")
            return optimized_text
        except Exception as e:
            logger.error(f"Error in text optimization: {e}")
            return None

    def generate_image(self, prompt: str, output_path: str = "generated_image.png") -> Optional[str]:
        """Generate image from text using SiliconFlow API"""
        url = f"{self.base_url}/images/generations"
        logger.info("Starting image generation")
        logger.debug(f"Prompt: {prompt}")
        
        payload = {
            "model": MODEL_IMAGE,
            "prompt": prompt,
            "image_size": "1024x1024",
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 15,
            "prompt_enhancement": False
        }

        try:
            logger.debug(f"Making POST request to {url}")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            logger.debug(f"Request headers: {json.dumps(self._get_masked_headers(), indent=2)}")  # Use masked headers for logging
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            # Log the response details
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            logger.debug(f"Response content: {response.text}")
            
            response.raise_for_status()
            
            # Save the image
            image_data = response.json().get('data', [{}])[0].get('url')
            if image_data:
                logger.info(f"Image generation successful, downloading from: {image_data}")
                image_response = requests.get(image_data)
                with open(output_path, 'wb') as f:
                    f.write(image_response.content)
                logger.info(f"Image saved to: {output_path}")
                return output_path
            else:
                logger.error("No image URL in response")
                return None
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return None

    def create_video(self, image_path: str, audio_path: str, output_path: str = "output.mp4"):
        """Create video from image and audio"""
        audio = None
        image = None
        final_video = None
        try:
            logger.info("🎬 Creating final video")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 加载音频
            logger.debug("🎵 Loading audio file...")
            audio = AudioFileClip(audio_path)
            
            # 创建图片剪辑
            logger.debug("🖼️ Loading image file...")
            image = ImageClip(image_path, duration=audio.duration)
            
            # 使用图片的原始尺寸创建视频
            logger.debug("🎞️ Compositing video...")
            final_video = CompositeVideoClip([image], size=(image.size))
            final_video.audio = audio
            
            # 写入输出文件
            logger.info("💾 Rendering final video...")
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium'
            )
            
            logger.info(f"✨ Video creation successful, saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error in video creation: {str(e)}", exc_info=True)
            return None
        finally:
            # 清理资源
            for clip in [audio, image, final_video]:
                if clip is not None:
                    try:
                        clip.close()
                    except Exception as e:
                        logger.error(f"⚠️ Error closing clip: {str(e)}")

    def process(self, audio_path: str, output_video_path: str = "output.mp4"):
        """Process the entire pipeline from audio to video"""
        # Step 1: Audio to text
        logger.info("Converting audio to text...")
        text = self.transcribe_audio(audio_path)
        if not text:
            return None
        
        # Step 2: Optimize text for image generation
        logger.info("Optimizing text for image generation...")
        optimized_text = self.optimize_text(text)
        if not optimized_text:
            return None
        
        # Step 3: Generate image
        logger.info("Generating image from text...")
        image_path = self.generate_image(optimized_text)
        if not image_path:
            return None
        
        # Step 4: Create video
        logger.info("Creating final video...")
        result = self.create_video(image_path, audio_path, output_video_path)
        
        if result:
            logger.info(f"Process completed successfully! Video saved to: {output_video_path}")
        return result

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert audio file to video with AI-generated image')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('--output', '-o', type=str, default='output.mp4',
                      help='Path to the output video file (default: output.mp4)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.audio_path):
        logger.error(f"Error: Audio file '{args.audio_path}' does not exist")
        return
    
    # Get API token from config
    processor = AudioToVideo(SILICON_FLOW_API_TOKEN)
    
    # Process the audio file
    result = processor.process(args.audio_path, args.output)
    
    if result:
        logger.info(f"Processing completed successfully! Output saved to: {args.output}")
    else:
        logger.error("Processing failed!")

if __name__ == "__main__":
    main()
