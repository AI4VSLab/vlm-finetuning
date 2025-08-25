from torch.utils.data import Dataset


class ChatWrapper(Dataset):
    def __init__(self, dataset, system_prompt):
        self.dataset = dataset
        self.system_prompt = system_prompt

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["instruction"]},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["response"]}],
            },
        ]
        return {"messages": conversation}

    def __len__(self):
        return len(self.dataset)
