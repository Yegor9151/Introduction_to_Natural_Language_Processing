import time
import tensorflow_datasets as tfds

def read_file(path: str, duration=False) -> str:
    start = time.time()
    with open(file=path, mode='r', encoding='utf-8') as file:
        text = file.read()
    if duration:
    	print(f'in {round(time.time() - start)} sec.')
    return text

def split_by_subwords(corpus_generator, file_name=None):
    start = time.time()
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator=corpus_generator, 
        target_vocab_size=2**13)
    if file_name:
        tokenizer.save_to_file(file_name)
        
    print(f'in {round(time.time() - start)} sec.')
    return tokenizer