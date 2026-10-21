"""Use the current resume dispatcher within the isolated text-v2 root only."""

from contextlib import contextmanager

from srgc_research.dispatch.ready_dispatch import GemmaResumeFirst as OriginalResume
from srgc_research.dispatch.resume_drain import resume_worker

from ..gemma4 import worker


class GemmaTextResumeFirst(OriginalResume):
    model_label = "Gemma-4-12B-TEXT-V2"
    retry_revision = "gemma4-text-v2-resume-budget-v1"


@contextmanager
def resume_first_worker():
    with resume_worker(worker, GemmaTextResumeFirst,
                       pattern="gemma4-12b-text-{dataset}.json",
                       env_key="SRGC_GEMMA_PLANS", label="GEMMA TEXT V2",
                       result_model="gemma4_text"):
        yield
