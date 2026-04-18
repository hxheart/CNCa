import torch

class FeatureRegistry:
    def __init__(self):
        self.registry = {}
        self.dim = 0
    
    def get_all_features(self):
        return sorted(list(self.registry.values()), key=lambda f: f.idx)

    def feature(self, name, encoding = None):
        if not name in self.registry.keys(): 
            assert encoding is not None, "cannot create feature without specifying an encoding"
            self.registry[name] = Feature(name, *encoding, self)
        return self.registry[name]

    def feature_value(self, name, value):
        return FeatureValue(self.feature(name), value)

    def embed_features(self, features):
        all_features = torch.cat([
            f.embed().unsqueeze(0) for f in features
        ])
        is_queried = torch.any((all_features == -2), 0)
        res = all_features.max(axis=0).values
        res[is_queried] = -2
        return res

    def unembed_features(self, t):
        features = []
        for f in self.get_all_features():
            res = f.unembed(t)
            if res: features.append(res)
        return features

class Feature:
    def __init__(self, name, encoder, decoder, registry):
        self.name = name
        self.idx = registry.dim
        registry.dim += 1

        self.registry = registry

        self.encoder = encoder
        self.decoder = decoder

    def value(self, value):
        return FeatureValue(self, value)

    def unembed(self, t):
        if t[self.idx] == -1: return None
        else: return self.value(t[self.idx].item())

    def __repr__(self):
        return f"{self.name} [{self.idx}]"

    def __getstate__(self):
        d = dict(self.__dict__)
        if "encoder" in d.keys():
            del d["encoder"]
        if "decoder" in d.keys():
            del d["decoder"]
        return d
    def __setstate__(self, d):
        self.__dict__.update(d)

class FeatureValue:
    def __init__(self, feature, value):
        self.feature: Feature = feature
        self.value = int(value)

    def embed(self):
        x = torch.ones([self.feature.registry.dim], dtype=torch.long) * -1
        x[self.feature.idx] = self.value
        return x