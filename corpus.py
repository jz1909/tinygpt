import torch

class corpus:

    def __init__(self, dataset_address):

        self.text = self.import_data(dataset_address)
        self.tokenizer = tokenizer(self.text)
        self.dataset = self.parse_data(self.text)
        self.train_data, self.eval_data = self._batch_split(self.dataset)

    def import_data(self, address):
        with open(address, 'r') as f:
            text = f.read()

        return text


    def parse_data(self, text):

        data = torch.tensor(self.tokenizer.encode(text), dtype = torch.long)
        return data

    def _batch_split(self,dataset):
        n = int(0.9*len(dataset))

        train_ds = dataset[:n]
        eval_ds = dataset[n:]

        train_and_eval = [train_ds, eval_ds]
        return train_and_eval

    def get_batch(self, set_type, block_size, batch_size):
        """
        Block size is the length of the sequence, batch size is how many concurrent blocks to analyze
        """

        if set_type == 'train':
            dataset = self.train_data
        elif set_type == 'eval':
            dataset = self.eval_data

        ix = torch.randint(0, len(dataset)-block_size, (batch_size, ))
        x = torch.stack([dataset[i:i+block_size] for i in ix])
        y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])

        return x, y


class tokenizer:
    """
    Most basic type of tokenization for now, later swap with tiktoken or some other way of tokenization
    """

    def __init__(self, dataset):

        self.chars = sorted(list(set(dataset)))
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string:str) -> list[int]:
        return [self.string_to_int[s] for s in string]

    def decode(self, integers: list[int]) -> str:
        return "".join([self.int_to_string[i] for i in integers]) 














