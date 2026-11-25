import os
import sys
import unittest

tree = os.environ["TREE"]
sys.path.insert(0, os.path.join(tree, "test/registered/amd/accuracy/mi35x"))
import test_glm53_flash_eval_mi35x as t

t.GLM_53_FLASH_MODEL_PATH = os.environ.get(
    "GLM53_MODEL", "/shared_nfs/jacchang/models/GLM-5.3-Flash-46a050"
)

if __name__ == "__main__":
    print("sglang tree:", tree, flush=True)
    unittest.main(module=t, argv=["run_glm53_eval", "-v"])
