# Superseded method comparison

The measurements behind section 5 of `notebooks/05_task4_triplet_encoder.ipynb`.

Four embedding methods - Classical (HSV histogram + gradient orientations),
Task3-CNN (the multi-task classifier reused as a feature extractor), a
convolutional autoencoder, and a plain triplet network - were built, scored
under the same protocol as the encoder that replaced them, and then removed.
Their checkpoints are gone; these tables are the record.

`retrieval_standard.csv` / `retrieval_strict.csv`
    the gallery-queries protocol, with and without suppressing every other
    photograph of the query's product.
`control_attribute_comparison.csv`
    the same methods re-scored on `baseColour`, which none of them was trained
    on. The Triplet model's 92.33 on `articleType` was produced by training on
    the metric; on the control attribute the ranking inverts.
`deployment_comparison.csv`
    the honest protocol - unseen products against a catalogue-only index -
    including the encoder that superseded them.

Notebook 05 rewrites the files one directory up when it runs. These copies do
not move.
