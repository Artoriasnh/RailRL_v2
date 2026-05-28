# RailRL System / Architecture Figure Brief

figure_goal: Create a manuscript-ready system architecture diagram for the RailRL paper.

paper_claim: RailRL is an end-to-end railway route-setting decision-support framework that connects operational data acquisition, traceable storage, leak-safe MDP reconstruction, conservative offline RL, and evaluation/explanation outputs.

figure_type: System / Architecture diagram.

mode: image mode, implemented locally as a vector-style Python/matplotlib schematic for reliable text rendering.

panels:
- Single full-width architecture figure with six logical blocks.
- Left-to-right technical flow: inputs -> acquisition/provenance -> canonical research store -> MDP dataset builder -> offline RL training core -> evaluation, explanation, and decision support.
- Bottom governance rail: temporal split, schema tests, leak audit, training gates, baseline/counterfactual evidence.

must_keep_labels:
- RailRL System Architecture
- Operational inputs
- Acquisition and provenance service
- Canonical research store
- Decision dataset builder
- Offline RL training core
- Evaluation and explanation
- Decision-support interface
- temporal split
- leak audit
- training gates
- counterfactual evidence

data:
- No numeric measurements are required in this architecture figure.
- Do not report performance improvements or deployment benefits as measured facts.

style_constraints:
- White background, restrained journal palette, readable labels at double-column width.
- Avoid decorative imagery and avoid visual claims that imply a deployed safety-critical control loop.
- Do not include protected operational-reference details in the public architecture.

output_formats:
- SVG for manuscript editing workflows.
- PDF for LaTeX/Word insertion.
- TIFF at 600 dpi for journal submission.
- PNG preview for quick inspection.

verification_checklist:
- All public labels are readable in PNG preview.
- No visible protected-reference labels appear in SVG text.
- Arrows show data/decision flow without implying online control.
- Figure exports are regenerated from a reproducible script.
