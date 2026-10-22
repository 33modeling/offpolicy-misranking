"""Use the current resume dispatcher within the isolated IT root only."""

from contextlib import contextmanager

from srgc_research.dispatch.ready_dispatch import GemmaResumeFirst as OriginalResume
from srgc_research.dispatch.resume_drain import resume_worker

from ..gemma4 import worker


class GemmaITResumeFirst(OriginalResume):
    model_label = "Gemma-4-12B-IT"
    retry_revision = "gemma4-it-v1-resume-budget-v1"


@contextmanager
def resume_first_worker():
    with resume_worker(worker, GemmaITResumeFirst,
                       pattern="gemma4-12b-it-{dataset}.json",
                       env_key="SRGC_GEMMA_PLANS", label="GEMMA IT",
                       result_model="gemma4_it"):
        yield
