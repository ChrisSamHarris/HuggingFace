from transformers import pipeline

# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")
result = classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "important", "tablet", "computer"],
)
print(result)

# Multi-modal visual question answering
# sudo apt install -y tesseract-ocr
# pip install pytesseract
vqa = pipeline(model="impira/layoutlm-document-qa")
result = vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)
print(result)