# PAN2: candidate video package

Assessment date: 2026-11-15. Status: research; no Cozy implementation or inference qualification.

Task owner: Codex. Purpose: assess IFM/PAN2 as a supported Cozy video tool and model.
Branch: `docs/pan2-assessment`. Base: `origin/master` at
`e4b179bbd099f681658dc7ef5a059f484969cf94`. Worktree:
`/home/fidika/cozy/.worktrees/packages/pan2-assessment`.

## Initial recommendation

Evaluate PAN2 as an additional action-oriented video generator alongside MiniMax H3.
The published checkpoint serves text-to-video and image-to-video from one starting frame.
It is not yet a supported Cozy package. Implementation, output quality, memory and latency
must be established separately before a release claim.

The upstream model card describes a 29B diffusion backbone, Qwen3.5-9B text encoder and
Wan 2.2 VAE, producing 704 × 1248 video at 24 fps for up to 20 seconds. Its conceptual
reasoner rewrites a user's request into a structured timeline before diffusion; it is a
separate stage rather than reasoning performed by the diffusion backbone itself.
[Model card](https://huggingface.co/IFM/PAN2),
[inference workflow](https://github.com/ifm-ai/PAN2/blob/main/docs/inference.md).

The paper supplied with the request, *Critique of World Model*, provides the broader
Generative Latent Prediction framing. The checkpoint, model card and executable source
are the integration authorities.
[Paper](https://arxiv.org/pdf/2507.05169).

## Scope of this assessment

- Check public capabilities and license files against the model card.
- Inspect the inference code, component sizes and hardware assumptions.
- Compare the implementation with current Cozy package boundaries.
- Propose the smallest useful package and concrete output checks.

No model weights have been downloaded and no paid hardware has been rented.
