from PIL import Image


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens for not being used in the loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch
