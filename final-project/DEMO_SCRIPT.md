# EcoSort Demo Video Storyboard

**Target Length**: 10 minutes  
**Purpose**: Demonstrate the EcoSort intelligent recycling sorting system as a complete product, showing the hardware setup, software interface, and end-to-end workflow.  
**Audience**: ICT304 assessor, potential stakeholders, general public.  
**Tone**: Professional, clear, enthusiastic.

---

## [0:00-0:30] Opening & Problem Statement
- **Visual**: Opening shot of a recycling bin with contaminated items (plastic bag in paper bin, food waste in glass, etc.). Slow zoom on a confused person looking at the bins.
- **Narration**:  
  "Recycling contamination is a costly problem. When non-recyclable items mix with recyclables, entire batches can be rejected and sent to landfill. At the point of disposal, people often guess which bin to use, leading to mistakes."
- **On-screen text**:  
  "Contamination → Rejected batches → Lost recyclables → Increased landfill"
- **Cut to**: Diagram of the EcoSort system: camera → AI classifier → decision routing → bin actuation → logging → dashboard.

---

## [0:30-1:30] System Overview & Components
- **Visual**: Animated exploded view of the EcoSort station (or photos of the physical prototype if built). Highlight each sub-system.
- **Narration**:  
  "EcoSort solves this by combining computer vision, intelligent routing, and real-time feedback into one integrated station. Let's look at the seven sub-systems that work together:"
  1. **Image Capture** – USB camera or smartphone captures a photo of the item.
  2. **Pre-processing** – Resizes and normalizes the image for the AI models.
  3. **Waste Classifier (AI)** – Two models: a lightweight SVM on hand-crafted features, and a state-of-the-art MobileNetV2 transfer learning model. Both output a class label and confidence score.
  4. **Decision & Bin-Routing** – Uses a confidence threshold θ to decide: if confidence ≥ θ, item goes to the matching bin; if below θ, item goes to a human-review queue.
  5. **Logging** – Every decision is recorded as a LogEntry (item ID, predicted class, confidence, routed to, timestamp, true class if known).
  6. **Dashboard** – Live statistics aggregated from the log: per-class counts, contamination rate (when ground truth is known), low-confidence rate (live workload signal).
  7. **Simulated Actuation** – Indicates which bin flap should open (or shows the review queue signal).
- **On-screen text**: Sub-system names and icons as they appear.
- **Cut to**: Close-up of the Streamlit app running on a laptop or tablet.

---

## [1:30-3:00] Software Interface Walkthrough
- **Visual**: Screen capture of the Streamlit app (navigating through tabs).
- **Narration**:  
  "The user interface is built with Streamlit, making it easy to interact with the system and view its internal state."
  - **Tab 1: Sort an item**  
    Demonstrate:  
    a) Upload a photo of a metal can.  
    b) Show the prediction: the class and its confidence, for example "Metal, 85%".  
    c) Show the routing decision: "Confidence 85% meets the 60% threshold, so the item goes to the Metal bin."  
    d) Show simulated actuation: green light on "Metal bin" label, action "Open metal flap".  
    e) Show the probability bar chart: metal highest, glass second.  
    f) Show the latency: about 220 ms per item for the two-model blend.  
    g) Show log entry added.
  - **Tab 2: Live dashboard**  
    Demonstrate:  
    a) After several items, show the dashboard updating.  
    b) Highlight per-class counts (e.g., 5 metals, 3 papers).  
    c) Show low-confidence rate (e.g., 2/15 = 13.3%).  
    d) Show contamination rate (if ground truth known from held-out samples).  
    e) Show recent decisions table.  
    f) Show timeseries chart of cumulative accuracy and review rate.
  - **Tab 3: Batch evaluation**  
    Demonstrate:  
    a) Click "Run batch" with 40 held-out test items.  
    b) Show progress bar.  
    c) Show results: accuracy about 92%, review rate about 10%, latency about 220 ms.  
    d) Note: All items processed through the full pipeline—capture, pre-process, classify, route, actuate, log—exactly as in live operation.  
    e) Show that dashboard totals match log totals (requirement from test plan).
  - **Tab 4: About**  
    Briefly show the system description, dependencies, and model info.

---

## [3:00-5:00] End-to-End Demonstration (Live)
- **Visual**: Alternative angle showing a physical setup (webcam, laptop, printed bin labels, or simulated actuation LEDs). If a physical prototype is not available, use the webcam feed from the app and show the simulated actuation on screen.
- **Narration**:  
  "Now let's see EcoSort in action with a stream of items."
- **Demonstrate**:  
  1. Place a glass bottle in front of the camera.  
  2. App shows: true label = Glass, prediction = Glass, confidence above 60%, routing to the Glass bin.  
  3. Actuation: Green light on Glass bin, action "Open glass flap".  
  4. Log entry appears.  
  5. Dashboard updates: Glass count +1.  
  6. Repeat with a plastic bag (predicted Plastic, confidence above 60% → Plastic bin).  
  7. Repeat with a clear glass cup. The confidence is near 50%, so the item goes to Human review. This shows the safeguard.  
  8. Actuation: Red light on Human review queue, action "Divert to review tray, hold item".  
  9. Dashboard: Low-confidence rate increases.  
  10. Repeat with a metal can (high confidence → Metal bin).  
  11. Show the dashboard after 5-6 items: per-class counts, low-confidence rate, contamination rate (if ground truth known).  
- **On-screen text**: Key metrics updating in real time.  
- **Cut to**: Split screen showing the item, the app prediction, and the dashboard.

---

## [5:00-7:00] AI Model Comparison & Justification
- **Visual**: Side-by-side comparison of the two models: SVM feature extraction diagram vs. MobileNetV2 architecture diagram.
- **Narration**:  
  "EcoSort evaluates two classifier techniques to choose the best for the task."
  - **SVM on hand-crafted features**:  
    Interpretable, lightweight, fast. Uses HOG (shape), colour histogram, and GLCM texture.  
    Accuracy: 71.1%, Macro-F1: 70.3%.  
    Pros: Explainable decisions, runs on CPU, low memory.  
    Cons: Limited by hand-crafted features; may miss complex patterns.
  - **MobileNetV2 transfer learning**:  
    Leverages features learned from over a million ImageNet images.  
    Accuracy: 77.6%, Macro-F1: 74.4%.  
    Pros: Higher accuracy, especially on shape-and-texture ambiguous classes (glass vs plastic).  
    Cons: Larger model (~9 MB), requires TensorFlow (falls back to SVM if unavailable).  
  - **Why MobileNetV2 is chosen as primary**:  
    Leads by 6.5 accuracy points and 4.1 macro-F1 points on identical held-out data.  
    The two-model blend takes about 220 ms per item. This is well under the 1-second requirement.  
    The confidence score enables the human-review fallback, making the opaque model acceptable.
  - **Note**: The SVM remains as a fallback for hardware without TensorFlow, providing an interpretable baseline.
- **On-screen text**: Table of results, latency, model size.

---

## [7:00-8:30] System Integration & Test Plan
- **Visual**: Flowchart showing the end-to-end pipeline with data flowing from capture to dashboard.
- **Narration**:  
  "EcoSort is designed as a system, not just a model. Every sub-system has a clear role, and the integration is rigorously tested."
  - **Unit tests**: Each sub-system is tested in isolation (e.g., logger totals match dashboard, routing threshold behaviour).  
  - **Integration tests**: The full pipeline is run on held-out items to verify:  
    - Each item reaches the correct bin or review queue.  
    - Every decision is logged.  
    - Dashboard totals equal log totals (no drift).  
    - Latency ≤ 1 second per item.  
  - **Test results**:  
    - 100% of logged items appear on the dashboard.  
    - Routing follows the threshold correctly.  
    - Mean latency about 220 ms (well under the 1 s requirement).  
    - Log persists to CSV and reloads correctly.  
    - Reset clears the system.
- **On-screen text**: Test names and pass/fail icons (all pass).

---

## [8:30-9:30] Deployment, Maintenance & Future Work
- **Visual**: Photos of a recycling station, a potential installation site, or a concept rendering.
- **Narration**:  
  "EcoSort is ready for deployment in real-world settings such as public recycling stations, office buildings, or materials recovery facilities."
  - **Deployment plan**:  
    1. Install hardware: USB camera, mini PC (e.g., Raspberry Pi 4 or equivalent), display, bin actuation mechanism (solenoids or motors), and indicator lights.  
    2. Install software: Clone repository, set up virtual environment, run `./run.sh`.  
    3. Calibrate: Run the confidence threshold sweep on-site if lighting conditions differ.  
    4. Train: Optional—retain or fine-tune models with local data.  
    5. Go live: Monitor dashboard, empty review queue periodically.  
  - **Maintenance**:  
    - Periodically audit a sample of items to update ground truth and compute contamination rate.  
    - Retrain models quarterly with new data to adapt to changing waste streams.  
    - Backup logs monthly.  
  - **Future work**:  
    - Add test-time augmentation for small accuracy gains.  
    - Collect more trash images to stabilise the rare class.  
    - Fine-tune the top layers of the MobileNetV2 backbone.  
    - Integrate with actual bin actuation hardware (serially via GPIO or network).  
    - Expand to more classes (e.g., separate beverage cartons, compostable plastics).  
    - Mobile app version for household use.
- **On-screen text**: Deployment checklist, maintenance schedule, future work icons.

---

## [9:30-10:00] Closing & Call to Action
- **Visual**: Return to the opening shot of the contaminated bin, now overlaid with the EcoSort station sorting correctly. Fade to the EcoSort logo and project details.
- **Narration**:  
  "EcoSort demonstrates how AI can be applied to real-world sustainability challenges. By combining accurate classification, intelligent routing, and transparent feedback, we reduce contamination, increase recycling rates, and move toward a circular economy.  
  Thank you for watching. For more information, see the project report and user guide."
- **On-screen text**:  
  "EcoSort: Intelligent Recycling Sorting System  
  ICT304 Final Project · Murdoch University  
  Team: Nivetha, Robert, Waitun  
  Date: October 2026  
  GitHub: [repository-url]"
- **Fade out**.

---

## Notes on Production
- **Audio**: Use a clear, warm voiceover. Record in a quiet environment.  
- **Music**: Optional soft, uplifting background track (royalty-free) at low volume during narration; raise during transitions.  
- **Captions**: Include subtitles for accessibility.  
- **Branding**: Use the EcoSort logo (♻️) and consistent colour scheme (green #2E7D32 for confident/bin actions, red #C62828 for review actions).  
- **Assets**: Screen captures from the Streamlit app, photos of the hardware setup (if available), diagrams created with draw.io or similar, and charts exported from the training notebook.  
- **Editing**: Cut to maintain pace; aim for 4-5 seconds per shot on average.  
- **Call to action**: Encourage viewers to explore the code, run the demo, and suggest improvements.

--- 
End of Storyboard
