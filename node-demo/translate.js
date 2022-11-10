/*
Written by KrishPro @ KP

filename: 'translate.js'
*/

const {Tokenizer} = require('./tokenizer')

const readline = require("readline-sync");
const { InferenceSession, Tensor } = require('onnxruntime-node');

async function get_tokenizers(pair) {
    let vocab_urls = {};

    if (pair.slice().sort().every((v, i) => v == ['de', 'en'][i])) {
        vocab_urls = {
            'de': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-german/vocabs/vocab.de',
            'en': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-german/vocabs/vocab.en',
        }
    }
    
    else if (pair.slice().sort().every((v, i) => v == ['en', 'fr'][i])) {
        vocab_urls = {
            'fr': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-french/vocabs/vocab.fr',
            'en': 'https://raw.githubusercontent.com/KrishPro/machine-translation/english-french/vocabs/vocab.en',
        } 
    }

    else {
        console.error("Invalid pair")
    }

    let tokenizers = pair.map((p) => new Tokenizer(vocab_urls[p]))

    for (let i = 0; i < tokenizers.length; i++) {
        let tokenizer = tokenizers[i];
        await tokenizer.load_vocab()
        tokenizers[i] = tokenizer
    }

    return tokenizers
}

async function get_models(pair){
    let pair_name;

    if (pair.every((v,i) => v==['en','fr'][i])){pair_name='english-to-french'}
    else if (pair.every((v,i) => v==['en','de'][i])){pair_name='english-to-german'}
    else if (pair.every((v,i) => v==['de','en'][i])){pair_name='german-to-english'}
    else if (pair.every((v,i) => v==['fr','en'][i])){pair_name='french-to-english'}
    else {console.error("Invalid pair")}

    console.log("Loading encoder")
    let encoder_sess = await InferenceSession.create(await (await fetch('https://gitlab.com/KrishPro/trained-models/-/raw/main/ONNX/'+pair_name+'/encoder.onnx')).arrayBuffer());
    console.log("Loading decoder")
    let decoder_sess = await InferenceSession.create(await (await fetch('https://gitlab.com/KrishPro/trained-models/-/raw/main/ONNX/'+pair_name+'/decoder.onnx')).arrayBuffer());
    console.log("Model loaded")

    return [encoder_sess, decoder_sess]

}

async function translate(text, tokenizers, models) {

    let pad_token = 0, sos_token = 1, eos_token = 2, max_len = 128;

    src_ids = [sos_token].concat(tokenizers[0].encode(text), [eos_token])
    
    src_ids = src_ids.concat(Array(max_len-src_ids.length).fill(pad_token))
    
    src_pad_mask = new Tensor('bool', src_ids.map((i) => i == pad_token)).reshape([1, max_len])
    
    src_ids = new Tensor('int64', src_ids.map(BigInt)).reshape([1, max_len])

    src_encodings = (await models[0].run({'onnx::Equal_0': src_ids}))['691']

    tgt = [[sos_token]]

    for (let i = 2; i < max_len; i++) {
        tgt_padded = new Tensor('int64', tgt[0].concat(Array(max_len-tgt[0].length).fill(0)).map(BigInt)).reshape([1, max_len])
        
        decoder_out = (await models[1].run({'onnx::Equal_0': tgt_padded, 'onnx::Slice_1': src_encodings, 'onnx::NonZero_2': src_pad_mask}))['1235']
    
        pred = parseInt(decoder_out.data[0])

        tgt = [tgt[0].concat([pred])]

        if (pred == eos_token) {
            break
        }
        
    }

    return tokenizers[1].decode(tgt[0])
}

async function sample(){

    let src="en", tgt="fr";
    let sample_sentences = {
        "en": "Hello, This is a very long sentence",
        "de": "Hallo, das ist ein sehr langer Satz",
        "fr": "Bonjour, c'est une très longue phrase"
    }

    let start = (new Date()).getTime()

    console.log("Loading models...(Usually takes 50 to 60s)")
    let tokenizers = await get_tokenizers([src, tgt])
    
    let models = await get_models([src, tgt])
    
    console.log((new Date()).getTime() - start, "s")
    
    console.log()
    console.log("Translating", src, "to", tgt)
    console.log("> ", sample_sentences[src])
    console.log("< ", await translate(sample_sentences[src], tokenizers, models))
    console.log("= ", sample_sentences[tgt])
    
}

async function main() {
    let pair = [0, 0];
    let src_language_code = readline.question('Source Language (en|fr|de): ')
    if (src_language_code!=='en' && src_language_code!=='fr' && src_language_code !=='de') {throw RangeError("Source language must en or fr or de, but is " + src_language_code)}
    pair[0] = src_language_code
    
    if (src_language_code == "en") {
        let tgt_language_code = readline.question("Target Language (fr|de): ")
        if (tgt_language_code !== 'fr' && tgt_language_code !== 'de') {throw RangeError("Target language must be fr or de, but is " + tgt_language_code)}
        pair[1] = tgt_language_code
    }
    else {
        console.log("Target Language (en): en")
        pair[1] = "en"
    }
    
    console.log("Loading models...(Usually takes 50 to 60s)")
    let tokenizers = await get_tokenizers(pair)

    let models = await get_models(pair)

    console.log()
    
    src = readline.question("Enter src sentence> ")

    while (src != '.exit') {
        let tgt = await translate(src, tokenizers, models);
        console.log("=> ",tgt)
        console.log()
        src = readline.question("Enter src sentence> ")
    }
    
}
  

main();