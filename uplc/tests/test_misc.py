import unittest

from .. import *


class MiscTest(unittest.TestCase):
    def test_unpack_plutus_data(self):
        p = Program(
            "0.0.1",
            Apply(
                BuiltIn(BuiltInFun.UnConstrData),
                data_from_cbor(
                    bytes.fromhex(
                        "d8799fd8799fd8799fd8799f581ce3a0254c00994f731550f81239f12a60c9fd3ce9b9b191543152ec22ffd8799fd8799fd8799f581cb1bec305ddc80189dac8b628ee0adfbe5245c53b84e678ed7ec23d75ffffffff581ce3a0254c00994f731550f81239f12a60c9fd3ce9b9b191543152ec221b0000018bcfe56800d8799fd8799f4040ffd8799f581cdda5fdb1002f7389b33e036b6afee82a8189becb6cba852e8b79b4fb480014df1047454e53ffffffd8799fd87a801a00989680ffff"
                    )
                ),
            ),
        )
        # should not raise anything
        p.dumps()
        r = Machine(p).eval()
        # should not raise anything
        r.dumps()
        self.assertEqual(
            r,
            BuiltinPair(
                l_value=BuiltinInteger(value=0),
                r_value=BuiltinList(
                    values=[
                        PlutusConstr(
                            constructor=0,
                            fields=[
                                PlutusConstr(
                                    constructor=0,
                                    fields=[
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(
                                                    value=b'\xe3\xa0%L\x00\x99Os\x15P\xf8\x129\xf1*`\xc9\xfd<\xe9\xb9\xb1\x91T1R\xec"'
                                                )
                                            ],
                                        ),
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusConstr(
                                                    constructor=0,
                                                    fields=[
                                                        PlutusConstr(
                                                            constructor=0,
                                                            fields=[
                                                                PlutusByteString(
                                                                    value=b"\xb1\xbe\xc3\x05\xdd\xc8\x01\x89\xda\xc8\xb6(\xee\n\xdf\xbeRE\xc5;\x84\xe6x\xed~\xc2=u"
                                                                )
                                                            ],
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                PlutusByteString(
                                    value=b'\xe3\xa0%L\x00\x99Os\x15P\xf8\x129\xf1*`\xc9\xfd<\xe9\xb9\xb1\x91T1R\xec"'
                                ),
                                PlutusInteger(value=1700000000000),
                                PlutusConstr(
                                    constructor=0,
                                    fields=[
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(value=b""),
                                                PlutusByteString(value=b""),
                                            ],
                                        ),
                                        PlutusConstr(
                                            constructor=0,
                                            fields=[
                                                PlutusByteString(
                                                    value=b"\xdd\xa5\xfd\xb1\x00/s\x89\xb3>\x03kj\xfe\xe8*\x81\x89\xbe\xcbl\xba\x85.\x8by\xb4\xfb"
                                                ),
                                                PlutusByteString(
                                                    value=b"\x00\x14\xdf\x10GENS"
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        PlutusConstr(
                            constructor=0,
                            fields=[
                                PlutusConstr(constructor=1, fields=[]),
                                PlutusInteger(value=10000000),
                            ],
                        ),
                    ]
                ),
            ),
        )
