import os
import sys

from classifier import BaseClassifier
import utils

# Implement your classifier class here,
class MyClassifier(BaseClassifier):
    ...

# Set this function to return your classifier
def get_classifier():
    from classifier import LearnableELM
    return LearnableELM()

# Run
if __name__ == "__main__":
    with open(__file__, "r", encoding='utf-8') as f:
        source = f.read()

    if len(sys.argv) != 2:
        raise RuntimeError(
            f'Please specify unique name for your classifier, for example  `python {os.path.basename(__file__)} "Logistic regression"`')

    utils.run_and_save(classifier_fn=get_classifier, name=sys.argv[1], source=source)
