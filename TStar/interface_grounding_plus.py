import os
from typing import Dict, Optional, List
import openai
from PIL import Image
import re
# It is assumed that TStar.utilites defines the following functions:
# - encode_image_to_base64: converts a PIL.Image to a base64 string.
# - load_video_frames: loads a specified number of frames from a video.
from TStar.utilites import encode_image_to_base64, load_video_frames


class LlavaInterface:
    """
    Example: Encapsulate inference calls for the Llava model.
    The key is to expose a unified method inference(query, frames, **kwargs).
    """
    def __init__(self, model_path: str, model_base: Optional[str] = None):
        # Load the Llava model logic should be implemented here.
        self.model_path = model_path
        self.model_base = model_base
        print(f"[LlavaInterface] model_path={model_path}, model_base={model_base}")

    def inference(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        **kwargs
    ) -> str:
        """
        Expose a unified inference interface.
        
        Args:
            query: User input, which may contain text and <image> tags.
            frames: List of corresponding image frames.
            system_message: System prompt.
            temperature, top_p, num_beams, max_tokens: Other inference parameters.
        
        Returns:
            A string containing the inference result.
        """
        print("[LlavaInterface] Inference called with query:", query)
        print("[LlavaInterface] frames count:", len(frames) if frames else 0)
        # In a real scenario, call the Llava model for inference.
        return "Fake Response from LlavaInterface"

from typing import List, Optional
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda"
    ):
        """
        初始化 Qwen 模型和 processor。
        """
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def inference(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128
    ) -> str:
        """
        统一的推理接口，支持文本和图像混合输入。
        
        Args:
            query: 用户的查询文本，可以包含 <image> 标签。
            frames: 图像帧列表，与 <image> 标签顺序对应。
            max_new_tokens: 生成回答时的最大新 token 数量。
        
        Returns:
            模型生成的回答字符串。
        """
        # 构建消息列表，按照 <image> 标签将文本和图像交替插入
        messages = []
        content_list = []
        parts = query.split("<image>")
        for i, part in enumerate(parts):
            if part.strip():
                content_list.append({"type": "text", "text": part.strip()})
            if frames and i < len(frames):
                # 这里直接传入 PIL Image 对象，processor 会处理图像输入
                content_list.append({"type": "image", "image": frames[i]})
        if not content_list:
            content_list.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content_list})
        
        # 使用 processor 生成模板文本，并处理视觉信息
        text_template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 准备输入张量
        inputs = self.processor(
            text=[text_template],
            images=frames,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # 调用模型生成回答
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # 裁剪掉输入部分，保留生成的回答部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

class GPT4Interface:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the GPT-4 API client. The API key is read from the environment
        variable OPENAI_API_KEY if not provided.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        openai.api_key = self.api_key

    def _build_messages(self, system_message: str, user_content: List) -> List[Dict]:
        """
        Build the messages list required by the OpenAI API.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def _encode_frames(self, frames: List[Image.Image]) -> List[Dict]:
        """
        Encode image frames into Base64 formatted messages.
        """
        messages = []
        for i, frame in enumerate(frames):
            try:
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}",
                        "detail": "low"
                    }
                }
                messages.append(visual_context)
            except Exception as e:
                raise ValueError(f"Error encoding frame {i}: {str(e)}")
        return messages

    def inference_text_only(
        self,
        query: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API for text-only input.
        """
        messages = self._build_messages(system_message, query)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def _inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API with frames as context.
        """
        user_content = [{"type": "text", "text": query}]
        try:
            user_content.extend(self._encode_frames(frames))
        except ValueError as e:
            return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_qa(
        self,
        question: str,
        options: str,
        frames: Optional[List[Image.Image]] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Perform multiple-choice inference using GPT-4 API.
        
        Args:
            question: The question to answer.
            options: Multiple-choice options as a string.
            frames: Optional visual context.
        
        Returns:
            The selected option (e.g., A, B, C, D).
        """
        query = (
            f"Question: {question}\nOptions: {options}\n"
            "Answer with the letter corresponding to the best choice."
        )
        user_content = [{"type": "text", "text": query}]
        if frames:
            try:
                user_content.extend(self._encode_frames(frames))
            except ValueError as e:
                return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        A unified inference interface supporting mixed text and image inputs.
        The query may include <image> tags.
        """
        parts = query.split("<image>")
        user_content = []
        for i, part in enumerate(parts):
            if part.strip():
                user_content.append({"type": "text", "text": part.strip()})
            if i < len(frames):
                try:
                    frame_base64 = encode_image_to_base64(frames[i])
                    visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                        }
                    }
                    user_content.append(visual_context)
                except Exception as e:
                    return f"Error encoding frame {i}: {str(e)}"
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


class TStarUniversalGrounder:
    """
    Combines functionalities of TStarGrounder and TStarGPTGrounder.
    Allows switching between LlavaInterface and GPT4Interface via the backend parameter.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        gpt4_api_key: Optional[str] = None,
        num_frames: Optional[int] = 8,
    ):
        self.backend = model_name.lower()
        self.num_frames = num_frames
        if "llava" in self.backend:
            if not model_path:
                raise ValueError("Please provide model_path for LlavaInterface")
            self.VLM_model_interface = LlavaInterface(model_path=model_path, model_base=model_base)
        elif "gpt" in self.backend:
            self.VLM_model_interface = GPT4Interface(model=model_name, api_key=gpt4_api_key)
        else:
            raise ValueError("backend must be either 'llava' or 'gpt4'.")

    def inference_query_grounding(
        self,
        video_path: str,
        question: str,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Dict[str, List[str]]:
        """
        Identify target objects and cue objects from the video based on the question.
        
        Args:
            video_path: Path to the video file.
            question: The question.
            options: (Optional) multiple-choice options.
        
        Returns:
            A dictionary with two keys: target_objects and cue_objects.
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames)
        system_prompt = (
            "Here is a video:\n" + "\n".join(["<image>"] * len(frames)) +
            "\nHere is a question about the video:\n" +
            f"Question: {question}\n"
        )
        if len(options) > 1:
            system_prompt += f"Options: {options}\n"
        system_prompt += (
            "\nWhen answering this question about the video:\n"
            "1. Identify key objects that can locate the answer (list key objects, separated by commas).\n"
            "2. Identify cue objects that might be near the key objects and appear in the scenes (list cue objects, separated by commas).\n\n"
            "Provide your answer in two lines, listing the key objects and cue objects separated by commas."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) != 2:
            raise ValueError(f"Unexpected response format --> {response}")

        target_objects = [self.check_objects_str(obj) for obj in lines[0].split(",") if obj.strip()]
        cue_objects = [self.check_objects_str(obj) for obj in lines[1].split(",") if obj.strip()]
        return target_objects, cue_objects

    def check_objects_str(self, obj: str) -> str:
        """
        Process the object string to normalize object names by:
        - Lowercasing
        - Removing prefixes like "1. ", "2. ", "Key objects:"
        - Removing punctuation
        - Stripping extra whitespace
        """
        obj = obj.strip().lower()

        # Remove known prefixes (with optional whitespace)
        obj = re.sub(r"^(key objects|cue objects)?[:\-]?\s*", "", obj)
        obj = obj.replace("key objects: ", "").replace("cue objects: ", "").replace(": ", "")
        obj = re.sub(r"^[0-9]+\.\s*", "", obj)  # e.g., "1. "
        
        # Remove punctuation like periods, colons etc.
        obj = re.sub(r"[^\w\s-]", "", obj)  # Keep letters, numbers, space, hyphen

        return obj.strip()

    def inference_qa(
        self,
        frames: List[Image.Image],
        question: str,
        options: str,
        temperature: float = 0.2,
        max_tokens: int = 128
    ) -> str:
        """
        Perform multiple-choice inference and return the most likely option (e.g., A, B, C, D).
        """
        system_prompt = (
            "Select the best answer to the following multiple-choice question based on the video.\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n" +
            f"Options: {options}\n\n" +
            "Answer with the option's letter from the given choices directly."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=30
        )
        return response.strip()

    def inference_openend_qa(
        self,
        frames: List[Image.Image],
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Perform open-ended question answering based on the video.
        """
        system_prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n"
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.strip()


if __name__ == "__main__":
    # Test example.

    qwen = QwenInterface(model_name="./pretrained/Qwen2.5-VL-3B-Instruct")
    frames_fake = None
    print("\n=== Using GPT-4 backend ===")
    gpt4_grounder = TStarUniversalGrounder(
        model_name="gpt-4o",
        gpt4_api_key=None,
        num_frames=2
    )
    searchable_objects = gpt4_grounder.inference_query_grounding(
        video_path="dummy_video.mp4",
        question="What objects are in the video?"
    )
    print("GPT-4 Grounding Result:", searchable_objects)

    question_mc = "How many cats can be seen?\n"
    options_mc = "A) 0\nB) 1\nC) 2\nD) 3\n"
    answer_gpt4 = gpt4_grounder.inference_qa(frames_fake, question_mc, options_mc)
    print("GPT-4 QA Answer:", answer_gpt4)
