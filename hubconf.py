dependencies = ["torch", "faiss", "numpy", "scipy"]


from model import S2SMaskGIT


def s2smaskgit() -> S2SMaskGIT:
    return S2SMaskGIT.from_remote()
