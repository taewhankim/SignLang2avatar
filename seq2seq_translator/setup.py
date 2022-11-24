from setuptools import setup
import os


setup(
   name='trans',
   author='Hyunsoo Lee',
   author_email='hs24716@aistudio.co.kr',
   packages=['core'],  # would be the same as name
   dependency_links = ["https://download.pytorch.org/whl/torch_stable.html"],
   install_reqs = [
    "pandas",
    "konlpy",
    "Levenshtein",
    "nltk",
    "gluonnlp",
    "transformers",
    "sentencepiece",
    "munch",
    "einops",
    "loguru",
    "boto3",
    "sklearn",
    "mxnet",
    "openpyxl",
    "tensorboard",
    "jamo",
    "torch==1.9.1+cu111",
    "torchvision==0.10.1+cu111",
    "torchaudio==0.9.1"
   ]
)
