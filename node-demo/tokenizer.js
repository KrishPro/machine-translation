/*
Written by KrishPro @ KP

filename: 'tokenizer.js'
*/

var unidecode = require('unidecode');

class Tokenizer{
    constructor (vocab_path) {
        this.vocab_path = vocab_path
        this.special_tokens = [0, 1, 2]
    }
    
    async load_vocab(){
        this.vocab = (await (await fetch(this.vocab_path)).json())['model']['vocab']
    }

    get_combs(word){
        let combs = [...Array(word.length+1).keys()].slice(1, word.length+1).map((i)=>word.slice(0, i))
        if (word.length == 0){
            return combs
        }
        combs = combs.concat(this.get_combs(word.slice(1, word.length)))
        return combs.sort((a, b)=> b.length-a.length)
    }

    split_into_tokens(text) {
        if (text in this.vocab) { return [text] }
        if (text == "") { return [] }

        let combs = this.get_combs(text)

        for (let i = 0; i < combs.length; i++) {
            let comb = combs[i];
            
            if (comb in this.vocab) {
                let before = this.split_into_tokens(text.slice(0, text.indexOf(comb)))
                let after =  this.split_into_tokens(text.slice(text.indexOf(comb)+comb.length, text.length))
                
                return before.concat([comb], after)
            }
        }
    }

    tokenize(text) {
        let tokenized = []
        let words = unidecode(text).split(" ")

        for (let i = 0; i < words.length; i++) {
            let word = words[i]
            
            tokenized = tokenized.concat(this.split_into_tokens(word))
        }

        return tokenized
    }

    encode(text) {
        let tokens = this.tokenize(text)
        return tokens.map((tok) => this.vocab[tok])
    }

    decode(tokens) {
        return tokens.filter((tok) => !(tok in this.special_tokens)).map((tok) => Object.keys(this.vocab).find(key => this.vocab[key] == tok)).join(" ")
    }
}

module.exports = {Tokenizer}