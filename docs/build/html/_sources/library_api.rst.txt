.. _Library_API:

Library API
===========

This page contains information about the classes in CtrlF-TF.

AlignParameters
---------------
.. autoclass:: ctrlf_tf.AlignParameters
   :members: save_parameters, from_parameter_file

BedTuple
--------
.. autoclass:: ctrlf_tf.site_call_utils.BedTuple

ClassifiedSequences
-------------------
.. autoclass:: ctrlf_tf.ClassifiedSequences
   :members: classify_from_dataframe, load_from_file, save_to_file

CtrlF
-----
.. autoclass:: ctrlf_tf.CtrlF
   :members: from_parameters, from_alignment_file, from_compiled_sites, save_alignment, save_compiled_sites, call_sites, call_sites_as_bed, call_sites_from_fasta

Optimize
--------
.. autoclass:: ctrlf_tf.Optimize
   :members: load_from_file, save_to_file

SiteTuple
---------
.. autoclass:: ctrlf_tf.site_call_utils.SiteTuple
