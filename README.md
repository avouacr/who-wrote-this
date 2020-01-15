# A RAMP starting kit for French novelists identification

*Authors : Romain Avouac, Jaime Costa Centena, Adrien Danel, Guillaume Desforges, José-Louis Imbert, Slimane Thabet*

Authorship identification is the task of recognizing who the author of a document is.
It is part of the Natural Language Processing (NLP) kind of tasks.

Being able to identify the author of a document presents several applications, such as detecting plagiarism or finding the author of an anonymous document.
Archives all around the world are full of documents for which knowing the author would be invaluable knowledge for historical studies.
Furthermore, [the multiple plagiarism scandals](https://lithub.com/12-literary-plagiarism-scandals-ranked/) in literature could be solved with an algorithm.
For instance the authorship of Moliere or Shakespeare has been debated from the 19th century (more on this [here](https://fr.wikipedia.org/wiki/Paternit%C3%A9_des_%C5%93uvres_de_Moli%C3%A8re) and [here](https://fr.wikipedia.org/wiki/Paternit%C3%A9_des_%C5%93uvres_de_Shakespeare)).

This task has also an instructive purpose.
It is a way to investigate if NLP algorithms are able to capture bot only the semantics, but also the literary style of a document.

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)


#### Local notebook

Get started on this RAMP with the [dedicated notebook](Project_French_author_classification.ipynb).

To test the starting-kit, run


```
ramp_test_submission --submission starting_kit
```

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.


