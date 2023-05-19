from typing import List
# from config import prompt_templates


class Prompt:
    """
    Generates a promt for every sentence according to the instructions provided
    """

    def __init__(self, language, task, num_prompts=3):
        self.language = language
        self.task = task
        self.dict_sa_prompt = prompt_templates[language][task]

    def __call__(self, sentences: List[str]) -> str:
        # :param sentences: a list with all the input sentences
        # :return: a string transformed to the desired prompt.
        raise NotImplementedError


prompt_templates = {"en": {"SA": {"active": ['Is this review positive? Yes or no? Review: {content} Answer:',
                                             'Do you consider this review positive? Yes or no? Review: {content} Answer:',
                                             'Did the user write a positive review? Yes or no? Review: {content} Answer:'
                                             ],
                                  "passive": ['Is it determined if this review is positive? Yes or no? Review: {content} Answer:',
                                              'Is this review considered positive by you? Yes or no? Review: {content} Answer:',
                                              'Was a positive review written by the user? Yes or no? Review: {content} Answer:'
                                              ],
                                  "auxiliary": ['Is this review considered positive? Yes or no? Review: {content} Answer:',
                                                'Shall this review be considered positive? Yes or no? Review: {content} Answer:',
                                                'Will this review be considered positive? Yes or no? Review: {content} Answer:'
                                                ],
                                  "modal": ['Can this review be considered positive? Yes or no? Review: {content} Answer:',
                                            'Could this review be considered positive? Yes or no? Review: {content} Answer:',
                                            'May this review be considered positive? Yes or no? Review: {content} Answer:'
                                            ],
                                  "rare_synonyms": ['Can this review be regarded as positive? Yes or no? Review: {content} Answer:',
                                                    'Can this review be assessed as positive? Yes or no? Review: {content} Answer:',
                                                    'Can this review be classified as positive? Yes or no? Review: {content} Answer:'
                                                    ]},

                           "NLI": {"active": ['Given {premise}, do we assume that {hypothesis} is true? Yes, no, or maybe?',
                                              'Take the following as truth: {premise}, does this truth entail the following statement: {hypothesis}? Yes, no or maybe?',
                                              'Given that {premise}, must it be true that {hypothesis}? Yes, no, or maybe?'
                                              ],
                                   "passive": ['Is {hypothesis} assumed to be true, given {premise}? Yes, no, or maybe?',
                                               'Is the following statement, {hypothesis}, entailed by the truth that {premise} is taken? Yes, no, or maybe?',
                                               'Is it necessary for {hypothesis} to be true, given that {premise}? Yes, no, or maybe?'
                                               ],
                                   "auxiliary": ['Given {premise}, do we assume that {hypothesis} is true? Yes, no, or maybe?',
                                                 'Given {premise}, shall we assume that {hypothesis} is true? Yes, no, or maybe?',
                                                 'Given {premise}, will we assume that {hypothesis} is true? Yes, no, or maybe?'
                                                 ],
                                   "modal": ['Given {premise} can we assume that {hypothesis} is true? Yes, no, or maybe?',
                                             'Given {premise} could we assume that {hypothesis} is true? Yes, no, or maybe?',
                                             'Given {premise} may we assume that {hypothesis} is true? Yes, no, or maybe?'
                                             ],
                                   "rare_synonyms": ['Given {premise} can we presume that “{hypothesis}” is true? Yes, no, or maybe?',
                                                     'Given {premise} can we postulate that “{hypothesis}” is true? Yes, no, or maybe?',
                                                     'Given {premise} can we conjecture that “{hypothesis}” is true? Yes, no, or maybe?'
                                                     ]}},
                    "de": {"SA": {"active": ['Ist diese Bewertung positiv? Ja oder nein? Bewertung: {content} Antwort:',
                                             'Betrachten Sie diese Bewertung als positiv? Ja oder nein? Bewertung: {content} Antwort:',
                                             'Hat der Benutzer eine positive Bewertung geschrieben? Ja oder nein? Bewertung: {content} Antwort:'
                                             ],
                                  "passive": ['Ist festgelegt, ob diese Bewertung positiv ist? Ja oder nein? Bewertung: {content} Antwort:',
                                              'Wird diese Bewertung von Ihnen als positiv betrachtet? Ja oder nein? Bewertung: {content} Antwort:',
                                              'Wurde eine positive Bewertung vom Benutzer verfasst? Ja oder nein? Bewertung: {content} Antwort:'
                                              ],
                                  "auxiliary": ['Wird diese Bewertung als positiv betrachtet? Ja oder nein? Bewertung: {content} Antwort:',
                                                'Soll diese Bewertung als positiv betrachtet werden? Ja oder nein? Bewertung: {content} Antwort:',
                                                'Wird diese Bewertung als positiv betrachtet werden? Ja oder nein? Bewertung: {content} Antwort:'
                                                ],
                                  "modal": ['Kann diese Bewertung als positiv betrachtet werden? Ja oder nein? Bewertung: {content} Antwort:',
                                            'Könnte diese Bewertung als positiv betrachtet werden? Ja oder nein? Bewertung: {content} Antwort:',
                                            'Dürfte diese Bewertung als positiv betrachtet werden? Ja oder nein? Bewertung: {content} Antwort:'
                                            ],
                                  "rare_synonyms": ['Kann diese Bewertung als positiv angesehen werden? Ja oder nein? Bewertung: {content} Antwort:',
                                                    'Kann diese Bewertung als positiv bewertet werden? Ja oder nein? Bewertung: {content} Antwort:',
                                                    'Kann diese Bewertung als positiv eingestuft werden? Ja oder nein? Bewertung: {content} Antwort:'
                                                    ]},

                           "NLI": {"active": ['Angesichts der {premise}, gehen wir davon aus, dass die {hypothesis} wahr ist? Ja, nein oder vielleicht?',
                                              'Nehmen Sie Folgendes als Wahrheit: {premise}, zieht diese Wahrheit die folgende Aussage nach sich: {hypothesis}? Ja, nein oder vielleicht?',
                                              'Vorausgesetzt, dass {premise}, muss es wahr sein, dass {hypothesis}? Ja, nein oder vielleicht'
                                              ],
                                   "passive": ['Wird angenommen, dass {hypothesis} wahr ist, vorausgesetzt {premise}? Ja, nein oder vielleicht?',
                                               'Wird die folgende Aussage, {hypothesis}, durch die Tatsache, dass {premise} akzeptiert wird, verursacht? Ja, nein oder vielleicht?',
                                               'Ist es notwendig, dass {hypothesis} wahr ist, vorausgesetzt {premise}? Ja, nein oder vielleicht?'
                                               ],
                                   "auxiliary": ['Angesichts der {premise}, nehmen wir an, dass {hypothesis} wahr ist? Ja, nein oder vielleicht?',
                                                 'Angesichts der {premise}, sollen wir annehmen, dass {hypothesis} wahr ist? Ja, nein oder vielleicht?',
                                                 'Angesichts der {premise}, werden wir annehmen, dass {hypothesis} wahr ist? Ja, nein oder vielleicht?'
                                                 ],
                                   "modal": ['Angesichts der {premise} können wir annehmen, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?',
                                             'Angesichts der {premise} könnten wir annehmen, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?',
                                             'Angesichts der {premise} dürfen wir annehmen, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?'
                                             ],
                                   "rare_synonyms": ['Angesichts der {premise} können wir davon ausgehen, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?',
                                                     'Angesichts der {premise} können wir postulieren, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?',
                                                     'Angesichts der {premise} können wir vermuten, dass "{hypothesis}" wahr ist? Ja, nein oder vielleicht?'
                                                     ]}}}
