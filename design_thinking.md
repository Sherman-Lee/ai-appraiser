# Core Design Philosophy: "Cognitive Offloading"
The objective is to transform the AI from a simple tool into a proxy agent. By allowing the AI to do the "looking," the survivor is protected from primary triggers while ensuring they receive the full compensation they are entitled to.

## Empathize
- **The User’s Reality:** Victims are navigating "loss," not just "claims." Every item represents a memory, and viewing photos of their former homes can trigger significant post-traumatic stress.

- **Key Insights:** Decision Fatigue: The cognitive load of remembering every item is overwhelming during a crisis.

- **Memory Block**: Trauma often hinders the ability to recall specific details required for high-accuracy appraisals.

- **Emotional Friction**: The direct visual confrontation with photos of a destroyed or lost home is the primary barrier to completing the task.

## Define
### Problem Statement:

"Displaced residents need a way to catalog lost belongings for insurance that minimizes direct exposure to traumatic imagery and automates the cognitive burden of valuation."

## Ideate
The AI serves as a buffer between the user and the trauma. Potential features include:

- **Privacy-First Image Scanning:** A system where the user uploads cloud backups, and the AI extracts item data in the background without requiring the user to view the images.

- **Purchase Receipts Support:** Automatic ingestion of receipts to reconstruct purchase history, including item names, prices, and dates—reducing reliance on memory.

- **Depreciation Consideration:** Calculate item depreciation (or appreciation, where applicable) based on category, age, and condition to provide realistic and insurance-aligned valuations.

## Prototype
A low-fidelity version of the app would focus on:

- **Low-Trigger UI:** A minimalist interface using only thumbnails and necessary details to describe items

- **Automated Valuation Engine:** A backend that automatically suggests Fair Market Value (FMV) for individual items and total inventory.

- **One-Click Import:** A simplified landing page for permission-based data ingestion from digital receipts and photo metadata.

- **Delimited File Export:** Easy export of structured inventory reports (e.g., CSV) formatted to meet insurance submission requirements.

## Test
Testing would be conducted with disaster recovery volunteers and survivor focus groups to measure:

- **Emotional Heat Maps:** Identifying if specific prompts or UI elements cause a spike in user anxiety.

- **Accuracy vs. Autonomy:** Ensuring the AI’s suggestions feel helpful rather than "taking over" the user's personal history.

- **Local Context:** Verifying the AI’s ability to recognize items common in the Altadena region (e.g., specific architectural styles or local high-end retailers).