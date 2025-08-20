import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

def check_gemma_structure():
    
    print("Loading Gemma 3 model...")
    
    # Hugging Face model id (same as in your code)
    model_id = "google/gemma-3-4b-pt"
    
    # Check GPU capability
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        print(f"GPU capability: {device_capability}")
        if device_capability[0] < 8:
            print("Warning: GPU doesn't support bfloat16, using float16 instead")
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
        print("CUDA not available, using CPU")
    
    # Define model loading arguments (minimal version of your config)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    # Add quantization to reduce memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )
    
    try:
        # Load the model
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        print("Model loaded successfully!")
        
        print("\n" + "="*50)
        print("SEARCHING FOR IMAGE/VISION COMPONENTS:")
        print("="*50)
        
        # Check for image/vision/siglip related modules
        vision_modules = []
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['vision', 'image', 'siglip']):
                vision_modules.append((name, type(module).__name__))
        
        if vision_modules:
            print("Found vision-related modules:")
            for name, module_type in vision_modules:
                print(f"  {name}: {module_type}")
        else:
            print("No vision modules found with keywords: vision, image, siglip")
        
        print("\n" + "="*50)
        print("TOP-LEVEL MODEL ATTRIBUTES:")
        print("="*50)
        
        # Check top-level attributes
        vision_attrs = []
        for attr in dir(model):
            if any(keyword in attr.lower() for keyword in ['vision', 'image', 'siglip']) and not attr.startswith('_'):
                vision_attrs.append(attr)
        
        if vision_attrs:
            print("Found vision-related attributes:")
            for attr in vision_attrs:
                attr_type = type(getattr(model, attr)).__name__
                print(f"  {attr}: {attr_type}")
        else:
            print("No vision attributes found")
        
        print("\n" + "="*50)
        print("ALL TOP-LEVEL ATTRIBUTES (first 10):")
        print("="*50)
        
        # Show first 10 non-private attributes for context
        all_attrs = [attr for attr in dir(model) if not attr.startswith('_')][:10]
        for attr in all_attrs:
            attr_type = type(getattr(model, attr)).__name__
            print(f"  {attr}: {attr_type}")
        
        print(f"\n... and {len([attr for attr in dir(model) if not attr.startswith('_')]) - 10} more attributes")
        
        print("\n" + "="*50)
        print("MODEL CONFIGURATION:")
        print("="*50)
        
        # Print model config for additional clues
        if hasattr(model, 'config'):
            config_dict = model.config.to_dict()
            vision_config_keys = [k for k in config_dict.keys() if any(keyword in k.lower() for keyword in ['vision', 'image', 'siglip'])]
            if vision_config_keys:
                print("Vision-related config keys:")
                for key in vision_config_keys:
                    print(f"  {key}: {config_dict[key]}")
            else:
                print("No vision-related keys found in config")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to insufficient GPU memory or missing dependencies")
        return
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print("Look for attributes like:")
    print("  • model.vision_model")
    print("  • model.vision_tower") 
    print("  • model.image_encoder")
    print("  • model.siglip_vision_model")
    print("Use the attribute names found above to replace the image encoder!")

if __name__ == "__main__":
    check_gemma_structure()