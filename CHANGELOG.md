# CHANGELOG

<!-- version list -->

## v0.9.0 (2026-07-09)

### Bug Fixes

- Corrected the links in CONTRIBUTING.md ([#305](https://github.com/Deltakit/deltakit/pull/305),
  [`d0856bc`](https://github.com/Deltakit/deltakit/commit/d0856bc25bc1671df53e57db496c959ba623a31c))

- Increase linkcheck_timeout to 180s to reduce flaky CI failures
  ([#291](https://github.com/Deltakit/deltakit/pull/291),
  [`f56fbcf`](https://github.com/Deltakit/deltakit/commit/f56fbcf38c618761ab109dd31edaa078f1fd28a6))

- **circuit**: Proper TICK placement with REPEAT
  ([#289](https://github.com/Deltakit/deltakit/pull/289),
  [`3f8b19d`](https://github.com/Deltakit/deltakit/commit/3f8b19dff8194f8a2a463d410f885f812ba915a9))

- **circuit**: Tag propagation in deltakit stim
  ([#302](https://github.com/Deltakit/deltakit/pull/302),
  [`dc4054a`](https://github.com/Deltakit/deltakit/commit/dc4054abb90c95d41a276d5d5894960a5a357cf4))

- **dependencies**: Deltakit-stim>=0.2 ([#296](https://github.com/Deltakit/deltakit/pull/296),
  [`7ba2de1`](https://github.com/Deltakit/deltakit/commit/7ba2de144a05be941278714f497b8641da080a42))

- **dependencies**: Update gql[requests] requirement from ~=3.5 to >=3.5,<5.0 in /deltakit-explorer
  in the prod-deps group across 1 directory ([#300](https://github.com/Deltakit/deltakit/pull/300),
  [`2f00bd1`](https://github.com/Deltakit/deltakit/commit/2f00bd1e8d20e6cd408f461308723c6ac746510a))

- **docs**: Add upper-bound for sphinx theme ([#280](https://github.com/Deltakit/deltakit/pull/280),
  [`76818c9`](https://github.com/Deltakit/deltakit/commit/76818c9a4b3cb96a08b8a69c04f6cb8ea7aee482))

- **docs**: Update release documentation ([#267](https://github.com/Deltakit/deltakit/pull/267),
  [`ead6b7f`](https://github.com/Deltakit/deltakit/commit/ead6b7f24b50317eb664a026d17ff5f42bb6bffc))

- **explorer**: Fix auth test. ([#288](https://github.com/Deltakit/deltakit/pull/288),
  [`9fad696`](https://github.com/Deltakit/deltakit/commit/9fad69613d5931113a4f851c7666e88c2250261b))

- **explorer**: Upper bound matplotlib version
  ([#285](https://github.com/Deltakit/deltakit/pull/285),
  [`de10516`](https://github.com/Deltakit/deltakit/commit/de10516c8d480937191323c4f98c85e15635879d))

- **other**: Move conftest.py file to analysis folder
  ([#255](https://github.com/Deltakit/deltakit/pull/255),
  [`f6cb203`](https://github.com/Deltakit/deltakit/commit/f6cb203a7785ab623d3f18623c71f980e3b39f78))

- **other**: Typo in documentation ([#254](https://github.com/Deltakit/deltakit/pull/254),
  [`63466b3`](https://github.com/Deltakit/deltakit/commit/63466b3d50f354942d7790d2f6a4607b31fa4a9e))

- **others**: Avoid coverage failing on dependabot PRs
  ([#256](https://github.com/Deltakit/deltakit/pull/256),
  [`60e8281`](https://github.com/Deltakit/deltakit/commit/60e8281c89f00b3399da43df668f28a668ee50fe))

### Features

- Allow even distances for lambda ([#242](https://github.com/Deltakit/deltakit/pull/242),
  [`1ed0ca3`](https://github.com/Deltakit/deltakit/commit/1ed0ca33d6ecfc7cd778f06fe0a2dc9ea6f3ccf4))

- Improve plotting function calls ([#245](https://github.com/Deltakit/deltakit/pull/245),
  [`80087d5`](https://github.com/Deltakit/deltakit/commit/80087d51bbdbdafc149d1eb3882d53febbcf2216))

- Lepprresults renaming ([#244](https://github.com/Deltakit/deltakit/pull/244),
  [`3e0c276`](https://github.com/Deltakit/deltakit/commit/3e0c2761e5f37c31584baeebe7134babbabff125))

- Refactor Generic Plot to Dispatch to Specialised Plotters
  ([#268](https://github.com/Deltakit/deltakit/pull/268),
  [`546117a`](https://github.com/Deltakit/deltakit/commit/546117ac34eb8b103d3da8cb131ac0b996de5c3f))

- Rename class to avoid name ambiguity. ([#243](https://github.com/Deltakit/deltakit/pull/243),
  [`a521905`](https://github.com/Deltakit/deltakit/commit/a521905fd9872c7a738e286ddc34165ccd613483))

- Replace stim imports ([#277](https://github.com/Deltakit/deltakit/pull/277),
  [`57126ab`](https://github.com/Deltakit/deltakit/commit/57126ab25cfe96b089ad0b90896de999cf7814b2))

- Visualise detection probability on patch (#135)
  ([#271](https://github.com/Deltakit/deltakit/pull/271),
  [`c79d4a0`](https://github.com/Deltakit/deltakit/commit/c79d4a0f45406de487e33cf1ac852b41b6ef2267))

- **analysis**: Support asymmetric error bars
  ([#275](https://github.com/Deltakit/deltakit/pull/275),
  [`26dae51`](https://github.com/Deltakit/deltakit/commit/26dae515059f07cabec0f6628c0c8864d44466b6))

- **analysis**: Use the uncertainties package for error propagation
  ([#274](https://github.com/Deltakit/deltakit/pull/274),
  [`6e3e31b`](https://github.com/Deltakit/deltakit/commit/6e3e31b58a91b4488da79cf59a9d6457e511f2d1))

- **explorer**: Implement c-optimal discretisation strategy to reduce gradient variance in error
  budgeting feature ([#266](https://github.com/Deltakit/deltakit/pull/266),
  [`f59bb4c`](https://github.com/Deltakit/deltakit/commit/f59bb4cd05a03f47d4b0effd0fc5f7a6fc704c1a))


## v0.8.0 (2026-03-09)

### Breaking Changes

- Support for Stim v1.11 is discontinued. Minimal supported Stim version is v1.12.

### Bug Fixes

- Do not use isinstance checks in dem parser ([#237](https://github.com/Deltakit/deltakit/pull/237),
  [`eed7123`](https://github.com/Deltakit/deltakit/commit/eed7123bf4c806574e61405de813a15644cac4a2))

- Numpy dimension issue in error budget ([#236](https://github.com/Deltakit/deltakit/pull/236),
  [`7b6eb37`](https://github.com/Deltakit/deltakit/commit/7b6eb3752c63b368385d7cf801dc8cac52deeb7c))

- Picklable Coordinate ([#239](https://github.com/Deltakit/deltakit/pull/239),
  [`d78fdc0`](https://github.com/Deltakit/deltakit/commit/d78fdc0ac270447fe87cf7a5ef9b8ad4792e9039))

### Documentation

- Document mutation in compilation functions ([#209](https://github.com/Deltakit/deltakit/pull/209),
  [`d559a1a`](https://github.com/Deltakit/deltakit/commit/d559a1a76e68debe1803d7dfee7eee2dba042b50))

### Features

- Generic Plotting Framework for Lambda and LEPPR Results
  ([#227](https://github.com/Deltakit/deltakit/pull/227),
  [`da36a8a`](https://github.com/Deltakit/deltakit/commit/da36a8a10bbb472cc69adc27662b426d0f69f4a6))

- Refactor draw patch ([#225](https://github.com/Deltakit/deltakit/pull/225),
  [`b555ea5`](https://github.com/Deltakit/deltakit/commit/b555ea5f24848acd8d8bbcea985e3844d7fdf663))


## v0.7.1 (2026-02-10)

### Features

- Fix dependencies of deltakit subpackages.


## v0.7.0 (2026-02-10)

### Features

- Implement error-budgeting ([#76](https://github.com/Deltakit/deltakit/pull/76),
  [`d624a98`](https://github.com/Deltakit/deltakit/commit/d624a98a156252a3ddeaf9bd69bec5a418d942af))


## v0.6.2 (2026-01-29)

### Bug Fixes

- Fix failing docs build on ReadTheDocs.

### Features

- Include deltakit-compile into release process.

## v0.6.1 (2026-01-29)

### Bug Fixes

- Make NoiseParameters fields keyword-only ([#194](https://github.com/Deltakit/deltakit/pull/194),
  [`ffc27b2`](https://github.com/Deltakit/deltakit/commit/ffc27b291de2035cfd4b9dd68649b8fb6235a6e4))

- Remove remaining references to pixi ([#195](https://github.com/Deltakit/deltakit/pull/195),
  [`1d8707d`](https://github.com/Deltakit/deltakit/commit/1d8707d4b50d2128d9e2402d97ab6aa3adc805e7))

### Features

- Deltakit compiler init ([#201](https://github.com/Deltakit/deltakit/pull/201),
  [`2441109`](https://github.com/Deltakit/deltakit/commit/2441109d543c8642d26d6874c35786f7f5b6650d))

- More re-exports in deltakit_core and bump patch version
  ([#200](https://github.com/Deltakit/deltakit/pull/200),
  [`0ba8885`](https://github.com/Deltakit/deltakit/commit/0ba888593f2e5a7b3314df7b729ff2124071278d))

- Update CONTRIBUTION and RELEASE ([#202](https://github.com/Deltakit/deltakit/pull/202),
  [`a259307`](https://github.com/Deltakit/deltakit/commit/a259307a57c42e05c434df12ae83aa2093a8ef4d))


## v0.5.1 (2025-09-17)

### Bug Fixes

- Add the Client class to the re-exports in deltakit
  ([#11](https://github.com/Deltakit/deltakit/pull/11),
  [`1df3056`](https://github.com/Deltakit/deltakit/commit/1df3056fd06bcb65052bb8adb82f31b69ffd5b46))

### Documentation

- Add Deltakit component images to README ([#8](https://github.com/Deltakit/deltakit/pull/8),
  [`5d41e7d`](https://github.com/Deltakit/deltakit/commit/5d41e7d6c8feb9800cf073d3524f317fe4084108))


## v0.5.0 (2025-09-16)

### Features

- Initial Public Release
