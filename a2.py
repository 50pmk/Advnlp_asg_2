from conllu import parse_incr, TokenList
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")


test_path = 'en_ewt-up-test.conllu'
train_path = 'en_ewt-up-train.conllu'


def extract_features(sentence: TokenList):
    # Use the tokens from the CoNLL file to ensure that we'll be working with the same set of tokens as
    # identified in the CoNLL file.
    conll_tokens = [token['form'] for token in sentence]
    doc = Doc(nlp.vocab, words=conll_tokens)

    # Process the doc with spaCy to add the linguistic features.
    doc = nlp.get_pipe("ner")(doc)
    # Initialize a list to store features for each token
    features = []
    for conll_token, spacy_token in zip(sentence, doc):
        # Create a dictionary for the  features
        token_features = {
            'text': conll_token['form'],  # Text from CoNLL
            'lemma': spacy_token.lemma_,
            'pos_tag': spacy_token.pos_,
            'dep_relation': spacy_token.dep_,
            'head_token': spacy_token.head.text,
            'morphology': spacy_token.morph,
            'is_alpha': spacy_token.is_alpha,
            'is_stop': spacy_token.is_stop,
            'is_punct': spacy_token.is_punct,
            'is_entity': spacy_token.ent_type_ != "",
            'entity_type': spacy_token.ent_type_
        }

        # Append the token's features to the list
        features.append(token_features)

    return features


def main():
    sentences: list[TokenList] = []
    with open(train_path, 'r', encoding='utf-8') as file:
        for sentence in parse_incr(file):
            features = extract_features(sentence)
            sentences.append(sentence)


if __name__ == '__main__':
    main()
