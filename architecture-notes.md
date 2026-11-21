# Room inference: training vs. inference

![Grouped architecture showing the README components and the training-only feedback loop](room-inference-architecture.svg)

The two README diagrams answer different questions: the first follows the data from WAVs to a predicted shell; the second shows how the model learns. The grouped diagram puts both on one page. Follow the top row for inference. During training, the lower row shows how the supervised and acoustic losses update the CNN.

The key difference is in **Intermediate calculation**: inference selects and orders walls using the CNN’s own predictions. Physics training uses the surveyed wall order to assemble the **predicted** planes. The simulator still runs on a predicted shell, so changing the CNN’s geometry can change the spectral error.
