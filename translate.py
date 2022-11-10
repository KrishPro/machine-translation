from tokenizers import Tokenizer
from unidecode import unidecode
import onnxruntime as ort
import numpy as np
import requests


def get_tokenizers(pair):
    vocab_urls = {}

    if sorted(pair) == ['de', 'en']:
        vocab_urls = {
            'de': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-german/vocabs/vocab.de',
            'en': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-german/vocabs/vocab.en',
        }

    
    elif sorted(pair) == ['en', 'fr']:
        vocab_urls = {
            'fr': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-french/vocabs/vocab.fr',
            'en': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-french/vocabs/vocab.en',
        } 
    
    tokenizers = [Tokenizer.from_str(requests.get(vocab_urls[p]).text) for p in pair]
    return tuple(tokenizers)

def get_models(pair):
    pair_name = ''

    if pair == ['en', 'fr']: pair_name='english-to-french'
    elif pair == ['en','de']: pair_name='english-to-german'
    elif pair == ['de','en']: pair_name='german-to-english'
    elif pair == ['fr','en']: pair_name='french-to-english'

    print("Loading models...(Usually takes 50 to 60s)")
    print("Loading encoder")
    encoder_sess = ort.InferenceSession(requests.get('https://gitlab.com/KrishPro/trained-models/-/raw/main/ONNX/'+pair_name+'/encoder.onnx').content)
    print("Loading decoder")
    decoder_sess = ort.InferenceSession(requests.get('https://gitlab.com/KrishPro/trained-models/-/raw/main/ONNX/'+pair_name+'/decoder.onnx').content)
    print("Model loaded")

    return encoder_sess, decoder_sess

def translate(text, src_tokenizer, tgt_tokenizer, encoder, decoder):

    pad_idx, sos_token, eos_token, max_len = 0, 1, 2, 128

    src_ids = src_tokenizer.encode(unidecode(text)).ids

    src_ids = np.expand_dims(np.array(src_ids + [0]*(max_len-len(src_ids))), 0)

    src_encodings = encoder.run(None, {'onnx::Equal_0': src_ids})[0]
    src_pad_mask = src_ids == pad_idx

    tgt = np.array([[sos_token]])
    for i in range(max_len-1):
        tgt_padded = np.concatenate([tgt, np.zeros((1, max_len-tgt.shape[1])).astype(np.int64)], axis=1)
        pred = decoder.run(None, {'onnx::Equal_0': tgt_padded, 'onnx::Slice_1': src_encodings, 'onnx::NonZero_2': src_pad_mask})[0]
        
        tgt = np.concatenate([tgt, np.expand_dims(pred, 0)], axis=1)

        if pred[0] == eos_token:
            break


    return tgt_tokenizer.decode(tgt[0])


def main():
  src_language_code = input("Enter src language (en|fr|de): ").strip().lower()
  assert src_language_code in ['en', 'fr', 'de'], f"Src must be 'en' or 'fr' or 'de', but got {src_language_code}"
  if src_language_code == "en":
    tgt_language_code = input("Enter tgt language (fr|de): ").strip().lower()
    assert tgt_language_code in ['de', 'fr'], f"Tgt must be 'fr' or 'de', but got {tgt_language_code}"
  else:
    tgt_language_code = "en"
    print("Enter tgt language (en): en")
  
  print()

  src_tokenizer, tgt_tokenizer = get_tokenizers([src_language_code, tgt_language_code])
  encoder, decoder = get_models([src_language_code, tgt_language_code])

  print()
  src = input("Enter src sentence> ")

  while src != "exit()":
    print(f"=> {translate(src, src_tokenizer, tgt_tokenizer, encoder, decoder)}")
    print()
    src = input("Enter src sentence> ")


main()