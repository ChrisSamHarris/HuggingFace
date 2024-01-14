from transformers import pipeline

CUDA_enabled = False

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
for i in preds:
    print(i)
    
if CUDA_enabled:
    def data():
        for i in range(1000):
            yield f"My example {i}"
            
    pipe = pipeline(model="gpt2", device=0)
    generated_characters = 0
    for out in pipe(data()):
        # AssertionError: Torch not compiled with CUDA enabled
        generated_characters += len(out[0]["generated_text"])
    
    
# The iterator data() yields each result, and the pipeline automatically recognizes the input is iterable and will start fetching 
# the data while it continues to process it on the GPU (this uses DataLoader(https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 
# under the hood). This is important because you don’t have to allocate memory for the whole dataset and you can feed the GPU as fast as possible.