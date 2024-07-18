# Captioning

This README provides literature overview and early proposal of captioning tool that provide the text description of the video.

/!\ The transcript is another text modality provided by the ASR tool in another module, independant from the caption /!\

## Goal

Multi-level hierarchical caption:
- Global summary of the full video: *description*
- Detailed description of scenes, at different temporal granularities:
 - *action caption* = event-level description of the main actions in the scene, every few frames
 - *background caption* = Scene-by-scene description of the background of the scenes, everytime the location / time of the action changes.

The audio is not a modality. But an [audio caption](https://arxiv.org/pdf/2403.15377) would be nice added modality (from a commonsense point of view).

Note that we can subdivise *semantic* granularity (main action, secondary action, background action, visual description of the main characters, or the basckground elements...) and *temporal granularity* (every movement, every action, every event, every scene...) as needed.

But as a first prototype, we want to get only the global description and a detailed description of each scene. 

## Pipeline

### Description

Option 1: Using vanilla video captioning. Note that many model can only handle short videos.

Option 2: Summarizing other textual resources (video metadata + transcript + scene by scene captions), LLM-only.


### Scene by scene caption

How to obtain them? There is no easy way, we have to compromise between the flexibility of the temporal granularity, the quality of the scene splitting, and the controllability of the captions. 

**Option 1:** 
- Split the video into short subsets using external tools like [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) to split into key scenes, 
- Use *vanilla video captioning* to describe the video subset, using different prompts.

Pro: Convenient, don't have to worry about wrongly splitting. Can use the power of the videoLLM to generate controllable, quality captions.

Con: Completely dependents on what the scene detection tool was trained on, so not flexible at all. Will not be able to use it if we want different temporal granularities. No temporal consistency, since the captions are independent from each other.

**Option 2:**
- Split the video using a rolling window of a few frames (depending on fps, but probably every 1-2 seconds).
- Use *vanilla video captioning* to describe the video subset, using different prompts.
- Use text similarity metrics and set a threshold to merge captions together.

Pro: Flexible, can do different splitting granularities depending on the desired level of detail and types of textual content.

Con: Computationally more expensive (two steps, and runs inference many times because of the rolling window). No temporal consistency, since the captions are independent from each other.

**Option 3:** Using a *dense captioning* model, that automatically does the splitting and the captioning.

Pro: Convenient, don't have to worry about wrongly splitting, fully end-to-end. More temporal consistency, since the caption benefits from the "memory" of the past of the video.

Con: Extremely limited, depends on the dense captioner's training data. No flexibility in the temporal granularity, the level of detail, the content of the caption.




# Models

## Vanilla video captioning
Mostly derived from existing VLMs. Some good ones are [PLLaVA](https://arxiv.org/pdf/2404.16994), [CogVLM2-video](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat), [Tasier](https://github.com/bytedance/tarsier).

PLLAVA is a good option:

```
git clone  https://github.com/magic-research/PLLaVA
# Follow README for installation. BUT missing ones, to install requirements:
pip install packaging
# to download model:
pip install huggingface_hub
# to run demo:
pip install opencv-python-headless
pip install numpy==1.26.3
pip install peft gradio decord einops moviepy imageio
pip install flash-attn --no-build-isolation
```
Here is a small script to run some captioning on a video (full or splitted), to add to the cloned repo:
``` tasks/eval/recaption/pllava_caption.py ```

Note: very good controlability, on the focus on the caption (background, actions, etc)



## [Dense captioning](https://paperswithcode.com/task/dense-video-captioning/latest)

Here what really matters, beyond the model quality, is what the training data looks like.

[VidChapters](https://github.com/antoyang/VidChapters) has a very large training dataset and has available checkpoints. See [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/9b5c3e00d6ed30aad7adac9e7a664de1-Paper-Datasets_and_Benchmarks.pdf). But it splits by "chapters", which seems to be very long scenes.

[VideoRecap](https://github.com/md-mohaiminul/VideoRecap): see [paper](https://arxiv.org/pdf/2402.13250).  
"augmenting Ego4D with long-range video summaries of hour-long videos. This leads to a hierarchical video captioning dataset consisting of short clip captions, medium-range segment descriptions, and long-range video summaries."

Follow installation script, very easy demo.


# Evaluation

What are the standard benchmarks for long video captioning? e.g. [MAS-QA, Ego-QA](https://arxiv.org/pdf/2405.19723)

How to compare the obtained captions on our own data, without gold captions?

- [AutoDQ](https://github.com/bytedance/tarsier) 

- GROOVIST (see groovist_improved.md)

- [ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP)