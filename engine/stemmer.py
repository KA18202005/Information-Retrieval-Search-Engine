"""
Porter Stemming Algorithm (1980)
Pure Python implementation of Martin Porter's classical suffix-stripping algorithm.
Preserved and integrated from local IR engine reference without external dependencies.
"""


class PorterStemmer:
    """Pure Python implementation of Martin Porter's 1980 Stemming Algorithm."""

    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def is_consonant(self, i: int) -> bool:
        if self.b[i] in 'aeiou':
            return False
        if self.b[i] == 'y':
            if i == self.k0:
                return True
            else:
                return not self.is_consonant(i - 1)
        return True

    def m(self) -> int:
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self.is_consonant(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self.is_consonant(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self.is_consonant(i):
                    break
                i += 1
            i += 1

    def vowel_in_stem(self) -> bool:
        for i in range(self.k0, self.j + 1):
            if not self.is_consonant(i):
                return True
        return False

    def double_consonant(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self.is_consonant(i)

    def cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self.is_consonant(i) or self.is_consonant(i - 1) or not self.is_consonant(i - 2):
            return False
        ch = self.b[i]
        if ch in ('w', 'x', 'y'):
            return False
        return True

    def ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        for i in range(length):
            if self.b[o + i] != s[i]:
                return False
        self.j = self.k - length
        return True

    def setto(self, s: str):
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length:]
        self.k = self.j + length

    def r(self, s: str):
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k -= 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k -= 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowel_in_stem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.double_consonant(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in ('l', 's', 'z'):
                    self.k += 1
            elif self.m() == 1 and self.cvc(self.k):
                self.setto("e")

    def step1c(self):
        if self.ends("y") and self.vowel_in_stem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def step2(self):
        actions = {
            'a': [("ational", "ate"), ("tional", "tion")],
            'c': [("enci", "ence"), ("anci", "ance")],
            'e': [("izer", "ize")],
            'l': [("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous")],
            'o': [("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            's': [("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous")],
            't': [("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            'g': [("logi", "log")]
        }
        if self.k > self.k0 + 1:
            penult = self.b[self.k - 1]
            if penult in actions:
                for suffix, replacement in actions[penult]:
                    if self.ends(suffix):
                        self.r(replacement)
                        break

    def step3(self):
        actions = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")
        ]
        for suffix, replacement in actions:
            if self.ends(suffix):
                self.r(replacement)
                break

    def step4(self):
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        ]
        for suffix in suffixes:
            if self.ends(suffix):
                if self.m() > 1:
                    self.k = self.j
                return
        if self.ends("ion"):
            if self.j >= self.k0 and self.b[self.j] in ('s', 't'):
                if self.m() > 1:
                    self.k = self.j

    def step5(self):
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self.double_consonant(self.k) and self.m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stems a single English word according to Porter's rules."""
        word = word.lower().strip()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[:self.k + 1]
