# Dataset and references

## RAVDESS

The audio comes from Steven R. Livingstone and Frank A. Russo's [RAVDESS dataset](https://zenodo.org/records/1188976). The [dataset paper](https://doi.org/10.1371/journal.pone.0196391) describes its construction and validation. These are acted expressions from 24 actors speaking two statements, so results should not be generalized to spontaneous emotion or other populations.

The audio-only collection contains 1,440 speech recordings and 1,012 song recordings. Song has fewer emotion classes and no recordings for actor 18. The current evaluation defaults to speech; mixing speech and song changes the task and class distribution. Actor IDs come from the recording filenames, not an inferred speaker-recognition model.

The record's explicit license section specifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), despite a shorter general attribution statement elsewhere on the page. That license requires attribution, noncommercial use and share-alike terms for distributed adaptations. The creators offer commercial licensing separately. This code repository does not grant rights to the dataset. Audio and feature tables stay local; check the applicable conditions before sharing derivatives or trained artifacts. No personal recordings are used.

Citation: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE 13(5): e0196391. DOI: [10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391).

## Research papers

These five papers arrived with the instructor's repository. I use links rather than redistributing the PDFs. The original local copies and their history remain in restricted recovery archives. This change to the current tree does not remove PDFs already present in the public fork's history.

| Paper | Citation link | Notice in the supplied PDF |
| --- | --- | --- |
| Head Fusion: Improving the Accuracy and Robustness of Speech Emotion Recognition on the IEMOCAP and RAVDESS Dataset | [10.1109/ACCESS.2021.3067460](https://doi.org/10.1109/ACCESS.2021.3067460) | CC BY 4.0. Redistribution has an express basis with attribution and the license conditions; linking is sufficient here. |
| ATTSF-Net: Attention-Based Similarity Fusion Network for Audio-Visual Emotion Recognition | [10.1109/TAFFC.2025.3591567](https://doi.org/10.1109/TAFFC.2025.3591567) | IEEE personal-use notice; redistribution requires permission. No permission established. |
| Spectro-Temporal Modulations Incorporated Two-Stream Robust Speech Emotion Recognition | [10.1109/TAFFC.2025.3531638](https://doi.org/10.1109/TAFFC.2025.3531638) | IEEE personal-use notice; redistribution requires permission. No permission established. |
| Lightweight Multimodal Emotion Recognition for Companion Robots: A Deep Learning Framework Integrating Facial and Speech Features | [10.1109/TAFFC.2025.3621086](https://doi.org/10.1109/TAFFC.2025.3621086) | IEEE personal-use notice; redistribution requires permission. No permission established. |
| Lightweight Spatio-Temporal Convolutional Neural Network for Audio-Visual Emotion Recognition | [10.1109/TAFFC.2025.3566773](https://doi.org/10.1109/TAFFC.2025.3566773) | IEEE personal-use notice; redistribution requires permission. No permission established. |

Four copies also carry institution-limited download notices. Their presence in a public course repository is not evidence of permission to republish them. DOI links above are transcribed from the documents; a license to the article does not automatically license the dataset or an unrelated implementation. This project does not reproduce these papers' deep-learning systems or claim their reported scores.

## Software and course materials

MiniLearn uses NumPy. The experiment tools use pandas, librosa and soundfile; scikit-learn supplies independent reference checks. Their respective upstream projects and licenses apply to those dependencies. The instructor's starter content has no source license in the refreshed upstream tree and is excluded from the clean candidate. See [provenance](PROVENANCE.md) for the exact division of supplied material and fork additions. Keep the candidate restricted until ownership/course publication confirmations and explicit publication approval are resolved. Aggregate result figures were generated from the actual cached experiments; no audio, feature vectors, saved models or per-record outputs are included.
