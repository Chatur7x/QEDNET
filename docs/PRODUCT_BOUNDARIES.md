# Product Boundaries: QEDNET Research and Clinical Decision Support

## QEDNET

QEDNET remains the research platform. Its purpose is to run and report reproducible research experiments comparing classical and simulator-based quantum ML on permitted, de-identified datasets. It must label estimates as research outputs, retain provenance, surface uncertainty, and avoid diagnostic or treatment language.

The present QEDNET cloud instance is a demo. The application now requires Supabase email/password authentication and a single allowlisted email before production routes are available, but the current cloud deployment does not have those settings configured. It has no tenant isolation, uses filesystem-demo mode, has no durable attached database/object store, and has no saved model runs. It must not receive patient data or be used for care decisions.

## Clinical decision-support product

Any clinical decision-support product is a separate product with separate source control, deployment, identity/access policies, data stores, audit trail, risk controls, model release process, and user-facing intended-use statement. Do not expose clinical workflows, patient identifiers, or clinical recommendations from this research application.

The clinical product requires its own intended-use definition, clinical partner and governance, representative external validation, human-factors work, security/privacy assessment, quality processes, and jurisdiction-specific regulatory review. These are organizational and clinical evidence requirements; a code change or benchmark result cannot complete them.

## Shared work

Research methods may be considered for later transfer only after independent review, provenance checks, compatibility review, and validation within the clinical product's own intended use. Shared branding, a shared login, or a copied model artifact does not establish clinical validation.

## Current agreed direction

- Keep this repository and its Render service scoped to QEDNET research.
- Design the clinical decision-support product as a separate application and release program.
- QEDNET's current preview uses verified email/password sign-in with one allowlisted email. Organization SSO can be restored later if the organization provides an OIDC identity provider.
- Free hosting is acceptable for a non-sensitive preview only. It does not provide the durable storage and operational guarantees required for a production research service that retains user uploads and experiments.
