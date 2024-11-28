import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat_stage2_lora import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
@dataclass
class CustomArguments:
    """
    Custom arguments for additional configurations like output files.
    """
    output_file: Optional[str] = field(
        default='results.csv',
        metadata={'help': 'The name of the CSV file to save evaluation results.'}
    )
    metrics_file: Optional[str] = field(
        default='metrics.txt',
        metadata={'help': 'The name of the TXT file to save evaluation metrics.'}
    )
    
@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )

class CustomTrainer(Trainer):
    def __init__(self, *args, output_file='results.csv', metrics_file='metrics.txt', **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float('-inf') 
        self.output_file = output_file
        self.metrics_file = metrics_file
    def save_lora_weights(self, model, save_directory):
        lora_state_dict = {}
        # 遍历模型的所有参数
        for name, param in model.named_parameters():
            # 检查参数名中是否包含 'lora'，以及参数是否需要梯度更新
            if 'lora' in name.lower() and param.requires_grad:
                lora_state_dict[name] = param.cpu().clone()
        if lora_state_dict:
            lora_save_path = os.path.join(save_directory, 'lora_weights.pth')
            torch.save(lora_state_dict, lora_save_path)
            logger.info(f'Saved LoRA weights to {lora_save_path}')
        else:
            logger.info('No LoRA weights found to save.')
    def evaluate(self, eval_dataset=None,ignore_keys=None):
        """
        Override the evaluation step to include your custom evaluation logic.
        """
        print('evaluate!')
        self.model.eval()
        eval_dataset = eval_dataset or self.eval_dataset
        from torch.utils.data import DataLoader
        if eval_dataset is None:
            raise ValueError("Evaluation dataset is None. Please provide a valid evaluation dataset.")
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1  # Use appropriate batch size
        )

        data_list = []

        # Loop through the evaluation dataset
        with torch.no_grad():
            for item2 in eval_dataloader:
                item = {k: v.to(self.model.device) for k, v in item2.items() if k != 'video_name' and k != 'answer'}
                video_name = item2['video_name'][0]
                answer = item2['answer'][0]

                # Perform a forward pass through the model
                output = self.model(
                    mos=item["mos"][0].to(torch.bfloat16),
                    pixel_values=item["pixel_values"][0].to(torch.bfloat16),
                    input_ids=item["input_ids"],
                    attention_mask=item["attention_mask"],
                    image_flags=item["image_flags"][0],
                    labels=item["labels"]
                )
                score1 = output['score1'].item()
                # Decode the result
                filtered_list = [x for x in output['label'] if x != -100 and x != 92542]
                decoded_output = self.tokenizer.decode(output['logit'][-len(filtered_list) - 1:-1])

                # Determine the quality level based on the decoded output
                if 'bad' in decoded_output:
                    level = 1
                elif 'poor' in decoded_output:
                    level = 2
                elif 'fair' in decoded_output:
                    level = 3
                elif 'good' in decoded_output:
                    level = 4
                elif 'excellent' in decoded_output:
                    level = 5
                else:
                    level = 0

                mos_score = item["mos"][0].item()

                # Store the result for evaluation
                data_list.append([video_name, answer, decoded_output, mos_score, score1, level])

        # Save the results and compute metrics
        # self.save_and_evaluate(data_list)
        metrics = self.save_and_evaluate(data_list, output_file=self.output_file, metrics_file=self.metrics_file)


        # Track best metric (replace 'accuracy' with the metric you are tracking, e.g., SRCC, PLCC, etc.)
        if metrics["SRCC_score"] > self.best_metric:
            self.best_metric = metrics["SRCC_score"]
            print(f"New best SRCC_score: {self.best_metric}. Saving model...")
            self.save_model(self.args.output_dir)  # Save the model's weights to output_dir
            self.save_lora_weights(self.model, self.args.output_dir)
    def save_and_evaluate(self, data_list, output_file='results.csv', metrics_file='metrics.txt'):
        """
        Function to save the evaluation results to a CSV file and compute metrics (SRCC, PLCC, KRCC),
        and save them to a .txt file.
        """
        # Convert the list into a pandas DataFrame
        df = pd.DataFrame(data_list, columns=['video_name', 'answer', 'output', 'mos', 'pred_score','level'])

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f'Results saved to {output_file}')

        total = len(data_list)
        right = 0
        for item in data_list:
            if item[2] in item[1]:
                right += 1
        accuracy = right / total
        print(f"Accuracy: {accuracy}")

        mos_values = df['mos']
        predicted_values = df['pred_score']
        srcc, _ = spearmanr(mos_values, predicted_values)
        plcc, _ = pearsonr(mos_values, predicted_values)
        krcc, _ = kendalltau(mos_values, predicted_values)
        # Print results
        print(f'SRCC_score: {srcc}')
        print(f'PLCC_score: {plcc}')
        print(f'KRCC_score: {krcc}')
        
        with open(metrics_file, 'a') as f:
            f.write(f"SRCC_score: {srcc}\n")
            f.write(f"PLCC_score: {plcc}\n")
            f.write(f"KRCC_score: {krcc}\n")
        
        # Calculate SRCC, PLCC, KRCC
        mos_values = df['mos']
        predicted_values = df['level']
        srcc2, _ = spearmanr(mos_values, predicted_values)
        plcc2, _ = pearsonr(mos_values, predicted_values)
        krcc2, _ = kendalltau(mos_values, predicted_values)

        # Print results
        print(f'SRCC_level: {srcc}')
        print(f'PLCC_level: {plcc}')
        print(f'KRCC_level: {krcc}')
        
        # Save metrics to a .txt file
        with open(metrics_file, 'a') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"SRCC_level: {srcc2}\n")
            f.write(f"PLCC_level: {plcc2}\n")
            f.write(f"KRCC_level: {krcc2}\n")
        
        metrics = {
            "accuracy": accuracy,
            "SRCC_score": srcc,
            "PLCC_score": plcc,
            "KRCC_score": krcc,
            "SRCC_level": srcc2,
            "PLCC_level": plcc2,
            "KRCC_level": krcc2
        }

        return metrics

        
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        train_eval,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.raw_data = []
        logger.info('Formatting inputs...Skip in lazy mode')
        # print('train_eval',train_eval)
        if train_eval=='train':
            print('in')
            #assert meta['annotation_train'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
            with open(meta['annotation_train'], 'r') as f:
                print('in2')
                # print('self.raw_data',self.raw_data)
                self.raw_data = f.readlines()
                # print('self.raw_data',self.raw_data)
                if repeat_time < 1:
                    # If repeat_time is less than 1, select a portion of the data
                    self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
                if repeat_time > 1:
                    assert isinstance(repeat_time, int)
                    # Repeat the list if repeat_time is greater than 1
                    self.raw_data = self.raw_data * repeat_time
        
        if train_eval=='eval':
            #assert meta['annotation_test'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
            with open(meta['annotation_test'], 'r') as f:
                self.raw_data = f.readlines()
                if repeat_time < 1:
                    # If repeat_time is less than 1, select a portion of the data
                    self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
                if repeat_time > 1:
                    assert isinstance(repeat_time, int)
                    # Repeat the list if repeat_time is greater than 1
                    self.raw_data = self.raw_data * repeat_time
                    
        # print('self.raw_data',self.raw_data)
        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                # if 'length' in data_item:
                token_length = 461
                    # token_length = data_item['length']  # Use precomputed length if available
                # else:
                #     # Compute token length using the tokenizer
                #     conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                #     str_length = len(conversations)
                #     if str_length not in self.conv2length:
                #         token_length = tokenizer(
                #             conversations, return_tensors='pt', padding=False, truncation=False,
                #         ).input_ids.size(1)
                #         self.conv2length[str_length] = token_length + num_image_token * (
                #                     max_dynamic_patch + use_thumbnail)
                #     else:
                #         token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def get_index(self, bound, fps, max_frame, first_idx, num_segments):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=8):
        from decord import VideoReader, cpu
        pixel_values_list = []
        if 'cogvideo' in video_path:
            video_path = video_path.split('.mp4')[0]+'.gif'
        if video_path.lower().endswith('.gif'):
            # Load GIF frames
            gif_frames = []
            with Image.open(video_path) as img:
                for frame in range(img.n_frames):
                    img.seek(frame)
                    gif_frames.append(img.copy().convert("RGB"))
            max_frame = len(gif_frames) - 1
            fps = 1  # Setting default FPS for GIFs

            # Process GIF frames based on indices
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            for frame_index in frame_indices:
                img = gif_frames[frame_index]
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values_list.append(img[0])

        else:
            # Use Decord for non-GIF files
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())

            # Process Decord frames based on indices
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values_list.append(img[0])

        return pixel_values_list
        # return img
    
    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_name = data_item['video']
        video_path = os.path.join(self.root, video_file)
        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
       
        image_list = self.load_video(video_path)
    
        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        special_tokens = special_tokens + '\nMotion Feature: <image>'
        # print(special_tokens)
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens)

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        
        num_patches = pixel_values.size(0) + 1

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        num_image_tokens[-1] = 1
      
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_patches)
        # print('pixel_values',pixel_values.shape)
        # Create the final return dictionary
        mos = torch.tensor(float(data_item['mos']/100), dtype=torch.bfloat16)
        answer = data_item['conversations'][1]['value']
        # Create the final return dictionary
        ret = dict(
            video_name = video_name,
            answer = answer,
            mos = mos,
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            # pixel_values2=pixel_values2,
            image_flags=torch.tensor([1] * (num_patches-1), dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0) + 1

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    train_eval,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],train_eval,
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset

def load_lora_weights(model, lora_weight_path):
    lora_state_dict = torch.load(lora_weight_path)
    model.load_state_dict(lora_state_dict, strict=False)
    logger.info(f'Loaded LoRA weights from {lora_weight_path}')

def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "steps"
    training_args.do_eval =True
    training_args.per_device_eval_batch_size =1
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
        
        # lora_weight_path = "/root/autodl-tmp/InternVL/internvl_chat/work_dirs/stage2/FETV_mos1_test1_stage2_lora_bl/lora_weights.pth"
        # load_lora_weights(model, lora_weight_path)  # Add this line to load LoRA weights

    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args, 'train', tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type)
    eval_dataset = build_datasets(
        data_args, 'eval', tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type)
    print('eval_dataset',eval_dataset)
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)
        _freeze_params(model.slowfast_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
        _freeze_params(model.motion_mlp)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()
    print('training_args',training_args)
    
    trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=concat_pad_data_collator,
    output_file=custom_args.output_file,  # Pass custom argument
    metrics_file=custom_args.metrics_file  # Pass custom argument
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=None,
    #     tokenizer=tokenizer,
    #     data_collator=concat_pad_data_collator
    # )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # metrics = train_result.metrics
        # try:
        #     metrics['train_samples'] = len(train_dataset)
        # except:
        #     metrics['train_samples'] = -1

        # trainer.log_metrics('train', metrics)
        # trainer.save_metrics('train', metrics)
        # trainer.save_state()


if __name__ == '__main__':
    main()