The German UD is converted from the content head version of the universal
dependency treebank v2.0 (legacy):

https://code.google.com/p/uni-dep-tb/

The README for that project is included here.

The German UD conforms to the UD guidelines, but there are some exceptions.

Universal POS tags were assigned manually, while LEMMA and XPOSTAG were
predicted by TreeTagger (first for release 1.4; see Changelog below).
Morphological features were assigned using rules based on the values of the
other columns (UPOSTAG, XPOSTAG, LEMMA, FORM, DEPREL). Gender, number and
case of nouns and their det/amod children are based on the (manual) syntactic
annotation, e.g. nsubj => nominative. They should have high precision but
lower recall because we did not add them where the context did not provide
enough clues (morphological analyzer / lexicon was not used at this stage).



=== Machine-readable metadata =================================================
Documentation status: stub
Data source: automatic
Data available since: UD v1.0
License: CC BY-NC-SA 3.0 US
Genre: news reviews wiki
Contributors: Petrov, Slav; Seeker, Wolfgang; McDonald, Ryan; Nivre, Joakim; Zeman, Daniel
Contact: zeman@ufal.mff.cuni.cz
===============================================================================
(Original treebank contributors: Quirmbach-Brundage, Yvonne; LaMontagne, Adam; Souček, Milan; Järvinen, Timo; Radici, Alessandra)



* Changelog

2016-08-21 Dan Zeman

Added sentence ids.
Added LEMMA and XPOSTAG predicted by TreeTagger with a German model supplied with the tagger and available in Treex
(http://ufal.mff.cuni.cz/treex, commit 50ad1fe0b9907ac382cbcda0a0f102602abc21a0). The UPOSTAGs from the original data
(assigned manually) were not modified. Some features were also added if they could be derived from the information
already present. Features that need a lexicon and/or disambiguation, such as Gender, Number and Case, have only been
added if they can be deduced from the (manually annotated) dependency structure, plus a few heuristics (e.g. form
equal to lemma often but not always means singular).

The work was done mainly using the HamleDT::DE::FixUD block, see
https://github.com/ufal/treex/blob/master/lib/Treex/Block/HamleDT/DE/FixUD.pm

2015-11-08 Wolfgang Seeker

Removed sentences from test due to overlap with dev
(sent-no. 6, 8, 79, 80, 88, 108, 109, 118, 152, 154, 164, 167, 190, 191, 195, 206, 215, 220, 229, 247, 295, 346, 451)
Removed sentences from dev due to overlap with train
(sent-no. 616)



###############################################################################
LEGACY README FILE BELOW
###############################################################################

===================================
Universal Dependency Treebanks v2.0
===================================

This directory contains treebanks for the following languages:
  English, German, French, Spanish, Swedish, Korean, Japanese, Indonesian,
  Brazilian Portuguese, Italian and Finnish.

A description of how the treebanks were generated can be found in:

  Universal Dependency Annotation for Multilingual Parsing
  Ryan McDonald, Joakim Nivre, Yvonne Quirmbach-Brundage, Yoav Goldberg,
  Dipanjan Das, Kuzman Ganchev, Keith Hall, Slav Petrov, Hao Zhang,
  Oscar Tackstrom, Claudia Bedini, Nuria Bertomeu Castello and Jungmee Lee
  Proceedings of ACL 2013

A more detailed description of each relation type in our harmonized scheme is
included in the file universal-guidelines.pdf.

Each treebank has been split into training, development and testing files.

Each file is formatted according to the CoNLL 2006/2007 guidelines:

  http://ilk.uvt.nl/conll/#dataformat

The treebank annotations use basic Stanford Style dependencies, modified
minimally to be sufficient for each language and be maximally consistent across
languages. The original English Stanford guidelines can be found here:

  http://nlp.stanford.edu/software/dependencies_manual.pdf


==============================================================================
Version 2.0 - What is new
==============================================================================

1. Added more data for German, Spanish and French.
2. Added Portuguese Brazilian, Indonesian, Japanese, Italian and Finnish.
3. New content-head versions for 5 languages (see below).
4. A number of bug fixes in the harmonization process.

=====================
Standard dependencies
=====================

In release 2.0 we include two sets of dependencies. The first is standard
Stanford dependencies, which correspond roughly to the output of the
Stanford converter for English with the copula as head set to true. In
general, these are content-head dependency representations with two major
exceptions: 1) adpositions are the head in adpositional phrases, and 2) copular
verbs are the head in copluar constructions.

This data is in the std/ directory and contains all languages but Finnish.

Version 1.0 of the data is only standard.

==========================
Content head dependencies
==========================

In order to converge to a more uniform multilingual standard, in particular
for morphologically rich languages, this release also includes a beta version
of content-head dependencies for five languages: German, Spanish, Finnish,
French and Swedish. Here the content word is always the head of a phrase.

=============================================================================
Language Specific Information
=============================================================================

====================
English dependencies
====================

Note that the English dependencies are based on the original Penn Treebank data
automatically converted with the Stanford Dependency Converter. Instructions for
how to do this with corresponding scripts are included in the English directory.

====================
Finnish dependencies
====================

Finnish data is in the ch/fi directory and was produced by researchers at
the University of Turku. In that directory there are specific README and
LICENSE files for that data. Two things to note. First, the Finnish data is
only content-head. This is due to difficulties in automatically converting the
data to standard format from its original annotations. Second, we have included
a test set in the release, but this is not the real test set, just a subset of
the training. The true test set for this data is blind (as per the wishes of
the researchers at Turku). The unannotated test data is included as well as
instructions for obtaining scores on predictions.

=============================================================================
Other Information
=============================================================================

================================
Fine-grained part-of-speech tags
================================

In the CoNLL file format there is a coarse part-of-speech tag field (4) and a
fine-grained part-of-speech tag field (5). In this data release, we use the
coarse field to store the normalized universal part-of-speech tags that are
consistent across languages. The fine-grained field contains potentially richer
part-of-speech information depending on the language, e.g., a richer tag
representation for clitics.

=========================
Licenses and terms-of-use
=========================

For the following languages

  German, Spanish, French, Indonesian, Italian, Japanese, Korean and Brazilian
  Portuguese

we will distinguish between two portions of the data.

1. The underlying text for sentences that were annotated. This data Google
   asserts no ownership over and no copyright over. Some or all of these
   sentences may be copyrighted in some jurisdictions.  Where copyrighted,
   Google collected these sentences under exceptions to copyright or implied
   license rights.  GOOGLE MAKES THEM AVAILABLE TO YOU 'AS IS', WITHOUT ANY
   WARRANTY OF ANY KIND, WHETHER EXPRESS OR IMPLIED.

2. The annotations -- part-of-speech tags and dependency annotations. These are
   made available under a CC BY-NC-SA 3.0 non commercial license. GOOGLE MAKES
   THEM AVAILABLE TO YOU 'AS IS', WITHOUT ANY WARRANTY OF ANY KIND, WHETHER
   EXPRESS OR IMPLIED. See attached LICENSE file for the text of CC BY-NC-SA.

Portions of the German data were sampled from the CoNLL 2006 Tiger Treebank
data. Hans Uszkoreit graciously gave permission to use the underlying
sentences in this data as part of this release.

For English, Italian, Finnish and Swedish, please see licences included in
these directories or the following sources.

Finnish - http://bionlp.utu.fi/fintreebank.html
Swedish - http://stp.lingfil.uu.se/~nivre/swedish_treebank/
Italian - http://medialab.di.unipi.it/wiki/ISDT

We are greatful to researchers at those institutes who provided us data, in
particular:

Maria Simi and company from the University of Pisa.
  Converting Italian Treebanks: Towards an Italian Stanford Dependency Treebank
  Bosco, Cristina and Montemagni, Simonetta and Simi, Maria
  Proceedings of LAW VII \& ID

Filip Ginter and company from the University of Turku.
  Building the essential resources for Finnish: the Turku Dependency Treebank
  Haverinen, Katri and Nyblom, Jenna and Viljanen, Timo and Laippala,
  Veronika and Kohonen, Samuel and Missil{\"a}, Anna and Ojala, Stina and
  Salakoski, Tapio and Ginter, Filip
  Language Resources and Evaluation, 2013

Joakim Nivre and company from Uppsala University.

And Chris Manning and Marie-Catherine de Marneffe from Stanford and Ohio.
  Generating typed dependency parses from phrase structure parses
  MC De Marneffe, B MacCartney, CD Manning,
  Proceedings of LREC, 2006

Any use of the data should reference the above plus:

  Universal Dependency Annotation for Multilingual Parsing
  Ryan McDonald, Joakim Nivre, Yvonne Quirmbach-Brundage, Yoav Goldberg,
  Dipanjan Das, Kuzman Ganchev, Keith Hall, Slav Petrov, Hao Zhang,
  Oscar Tackstrom, Claudia Bedini, Nuria Bertomeu Castello and Jungmee Lee
  Proceedings of ACL 2013

=======
Contact
=======

ryanmcd@google.com
joakim.nivre@lingfil.uu.se
slav@google.com
