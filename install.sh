# Follow the below installation guideline. You can run install.sh directly after creating the environment

# create environment. Note that python=3.9
# conda create --name tstar python=3.9 # mmdet's forced dependency for YoloWorld
# conda activate tstar

# Optional. Do this if you want to load LLaVA-NeXT to understand videos; No need to do this if you want GPT-4 for VQA
# git clone https://github.com/LLaVA-VL/LLaVA-NeXT
# cd LLaVA-NeXT && pip install -e . && cd ..

# below may take 5-15 minutes
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World && pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 wheel && pip install -e . && cd ..

pip install -r requirements.txt 

# Fix mmdet/mmyolo related issues
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmdet').origin;print(filename)")
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmyolo').origin;print(filename)")
pip install --upgrade setuptools

# download model
cd ./pretrained/YOLO-World && wget https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth && cd ../..

# download data
mkdir -p data/coco/lvis
wget -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
mkdir -p data/texts
wget -O data/texts/lvis_v1_class_texts.json https://github.com/AILab-CVC/YOLO-World/raw/refs/heads/master/data/texts/lvis_v1_class_texts.json

python run_demo.py     --video_path ../kfs-train-clip/0a060760-c33f-4160-8719-25725b570043.mp4     --question "What color is my gloves?"     --options "A) Green\nB) Yellow\nC) Blue\nD) Brown\n"
