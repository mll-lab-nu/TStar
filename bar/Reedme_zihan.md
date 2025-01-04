Draft of our open-source project code:

Make sure it is user-friendly: easy to use, and well-documented. (Will let chatgpt give a draft for this.)

Main scripts and functions:
1. Running the model for inference with code
python tstar.py \
    --model mll-lab/tstar-v1 \
    --video example.mp4 \
    --query "What is the man look like?" \
    --output (default output.json. Contains model, video, query, keyframe-numbers, keyframe-jpg itself.)


2. python tstar.py \
    --model mll-lab/tstar-v1 \
    --dataset mll-lab/lv-haystack

 (and a brief introduction about the keys in the dataset, how to formulate it if the users wanted to, etc)

3. Running the model for inference with web-interface (please refer to oasis)
4. Training code - How to train a world-yolo for efficient (what is needed)




Structure:

t-star.github.io
    | requirements.txt
    | main.py
    | main.sh
    | readme.md


2. Try to make sure:
    | The readme is simple
    | The requirementx.txt is minimal
    | The starting script can be launched in one-line-of-code (incl. model and dataset download, etc),
        while also highly customizable with argumentation (e.g. dataset or video to analyze, query, etc)


The model's class:
**Please make sure the main function is simple and most of the functions are imported from well-organized files. This would be very helpful for the users to get started and use our approach**

# MAYBE a module/grounder.py
class TStarGrounder(nn.Module): # Do we need to assess the grounder?
    def __init__(self, grounder_engine="llava-7b", config: transformers.ModelConfig=None):
        self.module = AutoModel.from_pretrained(grounder_engine)
        self.tokenizer = AutoTokenizer.from_pretrained(grounder_engine)
        self.config = {
            "prompt_template": "Please write the query to make it visually groundable...{query}"
        }
    
    def ground(self, query): # maybe we should add the video here? or the first frames to send to the grounder?
        if self.api_based:
            return self.ground_api(query)
        """The first-step of TStar -- grounding the query to visually descriptive objects for later search""
        input_string = self.config["prompt_template"].format(query=query)
        input_tokens = self.toknizer(input_string)
        output_tokens = self.module.generate(**input_tokens, config=self.config.generation_config)
        output_string = self.tokenizer.decode(output_tokens)
        objects_grounded = self.post_process(output_string) # process the answer to be visually descriptive object dict / list
        return objects_grounded
    
    def post_process(self, grounded_query):
    """process the answer to be visually descriptive object dict / list"""
        pass
        # TO JINHUI: would you return a list or dictionary here?

    def ground_api(self, query):
        ... # I think we should support gpt-4 tstar here


class TStarYoloObjectDetector(nn.Module):
    def __init__(self, model_name_or_path):
        ...
    
    def detect(self, frame_grid, object):
        # return a matrix same to the size of the grid. showing each small frame's score to have such objects



### maybe a module/searcher.py
class TStarObjectDetector(nn.Module):
    """input: one frame. Output: 
    def __init__(self, architecture, model_name_or_path):
        ARCH2CLASS = {"yoloworld": TStarYoloObjectDetector, ...}
        self.module = ARCH2CLASS[architecture](model_name_or_path)
        pass

    def detect(self, frame_grid, object):
        # return a matrix same to the size of the grid. showing each small frame's score to have such objects
        return self.module.detect(frame_grid, object)

class VideoGrid:
    # I think the main thing to be put here is the images theirselves and the size



class TStarSearcher(nn.Module):
    def __init__(self):
        self.object_detector = TStarObjectDetector(architecture = 'yoloworld', model_name_or_path = "xxx/YoloWorld", config=ModelConfig(grid_size, etc)) # please also complete This.
        self.video_sampler = TStarSampler(init_ditribution = 'uniorm', update_strategy = 'spline-prop')
    
    def search(self, grounded_query, video):
        """please consider the IO problem. It is very very very important!!! I think we can use avi videos which does not need frame-by-frame decoding. Otherwise the users may think we are taking probably too much time"""
        searched_frames = []
        self.video_sampler.reset()    
        grid = None
        while ...:
            grid = self.video_sampler.sample(video, grid)
            scores = self.object_detector.detect(grid, object)
            self.video_sampler.update(scores)


### maybe a module/reasoner.py
class TStarReasoner(nn.Module):
    """input query, key frames, return the answer to the query"""


## MAYBE a module/tstar.py
class TStar(nn.Module):
    def __init__(self, pretrain_path_or_name="mll-lab/tstar-v1", config: transformers.ModelConfig=None):
        self.query_grounder = TStarGrounder()
        self.searcher = TStarSearcher()
        self.reasoner = TSearReasoner()

    def forward(self, video: Decord.VideoObject, query):
        """input the video and query and output key frames"""
        grounded_query = self.query_grounder.ground(query) ## receive textual query and ground them to visually descriptive objects
        searched_frames = self.searcher(grounded_query, video) ##
        output_answer = self.reasoner.reason(query, searched_frames) ##
        return output_answer




## MAYBE main.py # I think most user does not have the patience to ground the query theirselves. So I think the main function and script would be lightweight and mainly do the search process.
## I also think we can apply openai for the first and last process. It would be very friendly to users if the only requirement for them is an openai api key and a gpu that could support 110MB yolo-world model rather than some 72B llava.









Other: dataset-wise



We can mainly include: the dataset type, how-to-use(maybe forced-fownload if not)

from datasets import load_dataset
dataset = load_dataset("mll-lab/lv-haystack")
print(dataset)

batch inference:

def inference(model, dataset, video_dir): # I want to maximize fast adaptation here. Let me think.
    video_dir = "path/to/your/videos" # if not downloaded, auto-download it.
    dataset = load_dataset("mll-lab/lv-haystack")




How to adapt to other datasets:
    important keys for each instance:
    {"video_path(relative path uner video_dir)", "query": "xxxx"}, This will be able for our code to identify and will return key-frames.
