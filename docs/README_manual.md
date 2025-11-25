# Using GEMINI

The experiments using GEMINI doesn't rely on the code that manages Ollama, a new set of funtions to make API calls has been devised. The code implemting this functionality is located at `star_code/scripts/`
The Gemini pipeline can be devised in two main stages:
1. Batch creation (implementation in `gemini_batch_creation.py`)
2. Batch processing (Implementation in `gemini_batch_processing`)

The two stage are piped together in a pipeline specified in `pipeline_script.py`.

## The Modules
### Batch Creation
The stage tasked with generating the batch of API request that need to be send to GCloud or Vertex. The code handling this stage is the one in the 'gemini_batch_creation.py' module. More specifically in this phase the code does:
- The model and the model's config such as temperature, top-p, the maximum number of token in output etc. These settings are hard coded in the source file, therfore to change them you need o change the values of `DEFAULT_GEN_CONFIG_ENTRY` in the source file:
```python
DEFAULT_GEN_CONFIG_ENTRY = {
    "thinkingConfig": {
        "thinkingBudget": 0,  # Disable thinking
        "includeThoughts": True,  # For troublehshooting
    },
    "maxOutputTokens": 8_192,
    "seed": 6,
    "temperature": 0.1,
}
```
- It builds the prompt for each istance of the data set starting from the template provided as input, differentianting for the different task of `vqa` (VideoQA), `sgg` (Scene Graph Generation) and `gu` (Graph-based QA, i.e.);
- Bundles together config, prompts and images in the correct JSON format expect by the gemini API (For batch mode, but also as an REST request-- the latter means that the json request cannot be sent directly by Google AI's python SDK but need to be send as an REST request, look at the [docs](https://ai.google.dev/gemini-api/docs/) for more).

The batch cretion modules divides the request in N (specified as input) files, which can be send separetly as the input for a batch request allowing for parallel processing.

The files created by the Batch Creation module are JSONL files, where each lines corresponds to a single request.

### Batch Processing
Handles the API request to Google Cloud Gemini or Vertex. From how to load and send the batch files create previously, waiting for the processed response, appending the new reply for implementing the two-stage prompting used during the thesis to finally saving all the result in a single file, specified as output.

## How to run
To run the 
Download the repository:

```bash
git clone https://github.com/FabioLusha/video-neurosymbolic.git
```

Build the container with the Dockerfile in dev_container:
```bash
docker build \
    --build-arg USERNAME="$USER" \
    --build-arg USER_UID="$(id -u)" \
    -t "$USER"/pydev2
```

Insert your `GOOGLE_API_KEY` in your `privte.env` file, and check the config in `compose.yaml` for:
- the container name and container image, verify that is pointing at "$USER"/pydev2
- the volume mounting points, if you don't want it to mess with your HOME directory change the first relative mount
- the volume mounting points for the shared datasets in /multiverse/datasets/shared


Run the container with:
```bash
docker compose up -d
``` 

Now change the directory to star_code/scripts and insepct the `pipeline_script.py` which contains the scirpt to run the whole pipeline and the arguments with examples for different experiments category: `vqa`, `sgg` and `graph-understanding`. To execute the type of experiment you want just uncomment the experiment you want to execute and comment out the other part. The settings for each experiments are separting in sections clearly delimited by `#=====...` comment lines and each section is distinguishible by looking at the `task` variable, indicating the experiment the section is specifying. For example the Graph understading section is delimited as following:

```python
    # ===========================================================================

    task = "graph-understanding"

    input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"
    # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"

    # The file containing the STSG associated to each video or question
    stsg_file = "/megaverse/storage/lusha/valset_sgg/aggregated_final_sgg_gemini2.5flash_val_part2_OutTokens8192_20250913_09:25:00.jsonl"

    # set limit_n != None if you want limit the processing to the first `limit_n` instances of the dataset
    limit_n = None
    # in how many batch-files to divide the dataset
    n_chunks = 5
    # path to the user prompt template
    user_prompt = "../data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v3.txt"
    videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    # sampling rate
    fps = 1
    # maximum number of frames to extract
    max_frames = 64
    output_file = "tmp/test_output.jsonl"
    reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"

    # model name from the one provided by Gemini
    model_name = "gemini-2.5-flash"
    # ===========================================================================

```
When setting the parameters be careful on where you run the script because of the relative paths.

To run the script get a shell session inside the running container with:
```bash
docker exec -it "$USER"_pydev2 bash
```

And run the script without any arguments:


