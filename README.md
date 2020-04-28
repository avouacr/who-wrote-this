# A RAMP starting kit for French novelists identification

*Authors : Romain Avouac, Jaime Costa Centena, Adrien Danel, Guillaume Desforges, José-Louis Imbert, Slimane Thabet*


![Mosaïque des auteurs](https://raw.githubusercontent.com/avouacr/who-wrote-this/master/author_mosaic.jpg "Mosaïque des auteurs")

This project is our contribution to the final challenge of the DATACAMP course (M2 Data Science). The goal was to develop a machine learning algorithm which can perform authorship identification, i.e. to automatically recognize the author of a given document based on its writing. 

Being able to identify the author of a document presents several applications, such as detecting plagiarism or finding the author of an anonymous document.
Archives all around the world are full of documents for which knowing the author would be invaluable knowledge for historical studies.
Furthermore, [the multiple plagiarism scandals](https://lithub.com/12-literary-plagiarism-scandals-ranked/) in literature could be solved with an algorithm.
For instance the authorship of Moliere or Shakespeare has been debated from the 19th century (more on this [here](https://fr.wikipedia.org/wiki/Paternit%C3%A9_des_%C5%93uvres_de_Moli%C3%A8re) and [here](https://fr.wikipedia.org/wiki/Paternit%C3%A9_des_%C5%93uvres_de_Shakespeare)).

This task has also an instructive purpose.
It is a way to investigate if NLP algorithms are able to capture not only the semantics, but also the literary style of a document.

## Data

The idea of this challenge is to see if an algorithm can identify some literary style for determining the author. In the general case, it can be very easy to differentiate documents on the genre e.g between novels, poems, and plays. The same is true for the period of writing, one can easily differentiate 17th century and 20th century style. Therefore, we decided to select authors from the same literary genre and the same period : a selection of French novelists from the 19th century. 

The 19th century seems ideal for this classification task as its language is close enough to contemporary French (allowing therefore a simple use of pre-loaded dictionnaries, stopwords, etc...), studied in class and features coherent and indetifiable litterary movements (whereas the 20th century litterary landscape is much more scattered). Moreover, 19th century books are in the public domain.


The following authors were selected in a attempt to be representative of the different styles of novels written in the 19th century:

-  **Naturalism :**
     - Zola
     - Maupassant
     - Daudet
     
-  **Literary realism:**
     - Stendhal
     - Balzac
     - Flaubert
     
-  **Romanticism:**
     - Hugo
     - Dumas
     - Vigny
     
-  **Early science-fiction/avant garde:**
     - Verne

We therefore have a meaninful sample of three litterary movements, and an *outlier*, Jules Verne. This can provide interesting follow-up analyses based on the results of this challenge.



## Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)


## Local notebook

Get started on this RAMP with the [dedicated notebook](Project_French_author_classification.ipynb).

To test the starting-kit, run


```
ramp_test_submission --submission starting_kit
```

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.


