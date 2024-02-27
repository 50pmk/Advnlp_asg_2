import nltk
import spacy
import re
from spacy.tokens import Doc
from conllu import parse, parse_incr, TokenList

nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')


test_path = 'en_ewt-up-test.conllu'
train_path = 'en_ewt-up-train.conllu'


def secondary_tokens(token):
    return True if '.' in token else False

def comments(token):
    return True if token.startswith("#") else False

def extract_predicates_and_arguments(tokens):
    if len(tokens) >= 11:
        return tokens[10], tokens[11:]
    else:
        return '_', []
    
def remove_numbers_and_special_characters(input_string):
    pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, '', input_string)

def extract_features_from_conll_line(line, is_predicate):

    fields = line.strip().split()
    token = {
        'form': fields[1],       # Surface form of the word
        'lemma': fields[2],      # Lemma of the word
        'pos_tag': fields[3],    # Part-of-speech tag
        'dep_relation': fields[6],  # Dependency relation
        'head_token': fields[8],    # Head token
        'morphology': fields[5],    # Morphological features
        'is_alpha': fields[1].isalpha(),  # Check if the token is alphabetic
        'is_stop': False,  # Since we're not using spaCy, we cannot determine if it's a stop word
        'is_punct': fields[1].isalnum(),  # Check if the token is a punctuation mark
        'is_predicate': is_predicate
    }

    return token



def main():
     
    features = []
    labels = []
    
    with open(train_path, 'r', encoding='utf-8') as file:
        sentences = file.read().strip().split('\n\n')
        limited_sentences = sentences[:1000]
    
        
        for sentence in sentences:
            
            sentence_features = []
            sentence_labels = []
            
            for line in sentence.split('\n'):

                if line.startswith("# sent_id"):
                    sent_id = line.split("=")[1].strip()
                    
                if comments(line):
                    continue
                
                parts = line.split("\t")
                token_id = parts[0]
                
                if secondary_tokens(token_id):
                    continue
                    
                predicate_sense, arguments = extract_predicates_and_arguments(parts)
                
                if predicate_sense != '_':
                    is_predicate = True
                    
                else: 
                    is_predicate = False

                
                if arguments[0] != '_': 
                    sentence_labels.append(arguments[0])
                    
                else: 
                    sentence_labels.append('O')
                                    
                token_features = extract_features_from_conll_line(line, is_predicate)
                sentence_features.append(token_features)
                
            features.append(sentence_features)
            labels.append(sentence_labels)
            


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
