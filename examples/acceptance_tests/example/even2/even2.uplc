(program
  1.0.0
  [
    [
      (force
        [
          (force
            (force
              (force
                (force
                  (delay
                    (delay
                      (delay
                        (delay
                          (lam
                            f_7
                            [
                              [
                                (force
                                  (delay
                                    (lam
                                      by_1
                                      [
                                        (force
                                          (force
                                            (delay
                                              (delay
                                                (lam
                                                  f_2
                                                  [
                                                    (force
                                                      (delay
                                                        (lam s_1 [ s_1 s_1 ])
                                                      )
                                                    )
                                                    (lam
                                                      s_3
                                                      (lam
                                                        x_4
                                                        [
                                                          [
                                                            f_2
                                                            [
                                                              (force
                                                                (delay
                                                                  (lam
                                                                    s_1
                                                                    [ s_1 s_1 ]
                                                                  )
                                                                )
                                                              )
                                                              s_3
                                                            ]
                                                          ]
                                                          x_4
                                                        ]
                                                      )
                                                    )
                                                  ]
                                                )
                                              )
                                            )
                                          )
                                        )
                                        (lam
                                          rec_8
                                          (lam
                                            h_11
                                            (delay
                                              (lam
                                                fr_14
                                                [
                                                  (force
                                                    [
                                                      by_1
                                                      (delay
                                                        (lam
                                                          fq_16
                                                          [
                                                            (force
                                                              [ rec_8 h_11 ]
                                                            )
                                                            [
                                                              (force h_11) fq_16
                                                            ]
                                                          ]
                                                        )
                                                      )
                                                    ]
                                                  )
                                                  fr_14
                                                ]
                                              )
                                            )
                                          )
                                        )
                                      ]
                                    )
                                  )
                                )
                                (lam
                                  k_9
                                  (delay
                                    (lam
                                      h_12
                                      [
                                        [
                                          h_12
                                          (lam
                                            x_15
                                            [
                                              (force k_9)
                                              (lam
                                                f_0_13
                                                (lam f_1_14 [ f_0_13 x_15 ])
                                              )
                                            ]
                                          )
                                        ]
                                        (lam
                                          x_18
                                          [
                                            (force k_9)
                                            (lam
                                              f_0_16
                                              (lam f_1_17 [ f_1_17 x_18 ])
                                            )
                                          ]
                                        )
                                      ]
                                    )
                                  )
                                )
                              ]
                              f_7
                            ]
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
          (delay
            (lam
              choose_5
              (lam
                even_0
                (lam
                  odd_1
                  [
                    [
                      choose_5
                      (lam n_2 [ [ (force n_2) (con bool True) ] odd_1 ])
                    ]
                    (lam n_3 [ [ (force n_3) (con bool False) ] even_0 ])
                  ]
                )
              )
            )
          )
        ]
      )
      (lam arg_0_0 (lam arg_1_1 arg_0_0))
    ]
    [
      (lam n_0 (delay (lam z_2 (lam f_3 [ f_3 n_0 ]))))
      [
        (lam n_0 (delay (lam z_2 (lam f_3 [ f_3 n_0 ]))))
        (delay (lam z_1 (lam f_2 z_1)))
      ]
    ]
  ]
)
