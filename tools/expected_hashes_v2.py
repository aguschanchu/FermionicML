"""expected_hashes_v2 -- the fermionicml-notebooks-v2 pin constants.

Data only (no logic beyond loading the frozen lists and the __main__
self-check).  The v1 module notebook_release/build/expected_hashes.py is a
SEPARATE file and is not touched by anything here.

PROVENANCE (extraction 2026-09-11T18:47:24Z)
  (a) V2_FINAL_STATE / V2_STD_STATS / V2_CONFIG: transcribed below from the
      frozen pull table copy_lists/v2/checkpoints_v2.tsv, every final_state
      digest of which was verified against
      results_v2/registry/V2-CHECKPOINTS.json at freeze time and again at
      build time.
  (b) V2_REGISTRY_SNAPSHOT: the lane -> final_state sha map of the frozen
      registry snapshot copy_lists/v2/V2-CHECKPOINTS.snapshot.json
      (40 lanes), i.e. the WHOLE registry, not only the bundled roster.
  (c) DIST_FILE_SHA256: loaded AT IMPORT from dist_files.sha256 next to this
      file (bundle: tools/copy_lists/dist_files.sha256; repo:
      notebook_release/build/copy_lists/v2/dist_files.sha256) -- the per-file
      sha256 of the staged v2 dist.  Two paths are excluded because they
      would have to contain their own digest: ['tools/copy_lists/dist_files.sha256', 'tools/expected_hashes_v2.py'].

Self-check (`python expected_hashes_v2.py`): asserts (a) is a subset of (b)
lane by lane, reloads (c) and reports its size, and (when run inside or
beside a staged dist) re-hashes every (c) entry against the tree on disk.
"""
import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

SNAPSHOT_UTC = "2026-09-11T18:47:24Z"
N_REGISTRY_LANES = 40
SELF_REFERENTIAL = ['tools/copy_lists/dist_files.sha256', 'tools/expected_hashes_v2.py']

# ---------------------------------------------------------------- (a)
V2_FINAL_STATE = {
    'v1chk_std_thermal_random_s42':
        '9c72bf5d12038b820ac038d9fdc243d6fa0d45f68faf8dfcb6b7047862565735',
    'v2abl_mlp_s42':
        '1ba6645cdb1e362977f03a1fae90255ca4fa750fd7e66d8f1fd3be0f30f698db',
    'v2abl_neutralbias_s42':
        '805b4b7672deb44a1903593706018d2f11592f25e42baed742f4a5d63e1bca32',
    'v2abl_noembed_s42':
        '1ee6b58035dc7a950c347392c378c36f4b0f82af70259345eac0dc9918d60992',
    'v2abl_noreinject_s42':
        '3e529480d32306202af5087007ebf2e7d342d09cdeaa647f155d860447a9d5fb',
    'v2abl_noscatter_s42':
        'a3b66b9625d984c1f28b1203163dd9a30e3e0f475373f5b053f6b40099429055',
    'v2abl_rdm_s42':
        '2cf68c44dcf557765a763b621e2629a9997970b39594ac9474f0955fd887938a',
    'v2d16progorig_const_gs_s42':
        'd22ae7652a7550e77a922744108079d1a8736874cab5eef1b2fac292f70ad2da',
    'v2d16progorig_vect_gs_s42':
        'd1d3d27ef49055fd39ee4f650400b50f509e23134e260f95271e4587023e2db9',
    'v2d16progstd_const_gs_s42':
        '54d00e19a966c7df864c18d53bba984923a14cb7e112a9d4d2d4af543f8d7204',
    'v2d16progstd_vect_gs_s42':
        'fab7a27d67fd4f2b05c16e658e4ffa7eca28b2aedd481238c7fd5bc2dd1378ec',
    'v2gram_thermal_random_50_s42':
        '16be3ef09470e427b98928856dc7a6f986b0683403ab5df3c28594bda0129b9d',
    'v2gs_const_s42':
        '90bdb57439342c68dcdc75ae25e8f09293932b5bba1fadb060daf024d45d10f3',
    'v2lc_thermal_1e6_s42':
        '9c5d5964f25fa8a2b776151b7fb23f6bd153beb37c46d35c8338cde854f8ef40',
    'v2lc_thermal_2e6_s42':
        '650ea581753534bb1a23c5cb459a5a094e767628c0137da22ffc7e5020532ac9',
    'v2lc_thermal_2p5e5_s42':
        '1b956c1f74e40c8ccd1421d3f5390ee7adcfa40e9ada6e1d4524b999b06e12c3',
    'v2lc_thermal_5e5_s42':
        '0731dea3d167fbc74c8a2affe9280bf944f2764b04ebe012eeb0a4e60e03dace',
    'v2mlp_cap_s42':
        'e347e4a03208da37cd37bcd2ef9e299e2a53f4821389663108782cbbd866be52',
    'v2mlp_gs_random_s42':
        '9fcf5967c5e91e2a8c373a0eca095f000dc65ba406c453d13041d5d3ed2cb791',
    'v2ogn_mse_s42':
        '2d8e62c2fa06e2686659ec83b3e4dead553c69efe8c3e8c53c6c2c2b2a34459d',
    'v2ogn_noE_s42':
        '367f245b1b6e7ec2b853572d3b8c82bb6b28292b193dd4f141d1e8c39323ebca',
    'v2std_thermal_random_s42':
        'f5b7e4cc6d2f0bff6be996a9584310074784ef1405081ef65b6b560acc68c5ce',
    'v2std_thermal_random_s43':
        '64dbdb91bf87a445d80e098f0ef4b92debd13e06f470041ab9994cf4d128f0b0',
    'v2std_thermal_random_s44':
        '75a4e760b1b0eafe721982353d809f9a45eed27a27e6e9e7a72901a4f080c8c7',
    'v2stdgs_gram_s42':
        '1ab83f4a8ba803e7fff7f9e324bb5415372bf80b3589962f5bc35cfbf10c00c2',
    'v2stdgs_random_s42':
        'f5784b584d6f2aae0a9a3271c2cb21d7f0393a7334c35c70cdb7117784667896',
    'v2stdgs_twin_s42':
        '80323aab2e7936fdbd78f3859cbff6abd444caad12ef9e9a52810e03ba9f2c77',
}

V2_STD_STATS = {
    'v1chk_std_thermal_random_s42':
        'b15711d3254f84fbd545c88a93a7642d5b963fa1d0a4443f0aeb445c649cfa2d',
    'v2abl_neutralbias_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2abl_noembed_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2abl_noreinject_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2abl_noscatter_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2abl_rdm_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2d16progstd_const_gs_s42':
        'd883d4c18def9305137a5f4f827be4424c3bce1c5d95a2c33dc410cc5e83d019',
    'v2d16progstd_vect_gs_s42':
        'b1d6c2ead60b99ed00491bd41b054c33660d48d3fd1c43ffdd5ea8edb5c127f2',
    'v2gram_thermal_random_50_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2lc_thermal_1e6_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2lc_thermal_2e6_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2lc_thermal_2p5e5_s42':
        'a4747f935ab4793529522336325d4d22a947e88f9ceffdcf7b2462a2a1625550',
    'v2lc_thermal_5e5_s42':
        'b028c9c050f41070069fd7cc642d5f8a81a0d7c3579b2aff292941a2814c9d03',
    'v2mlp_cap_s42':
        '4c112c905dc29aef5834ca05a931066ba2b72c994aad5e0633e2c3e6818bd1f3',
    'v2ogn_mse_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2ogn_noE_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2std_thermal_random_s42':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2std_thermal_random_s43':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2std_thermal_random_s44':
        '162f4a807de72c21f264adf651ca613e01a1619d0fa8e2630c2de7eaa4060d5d',
    'v2stdgs_gram_s42':
        'e3c33b918007aeca5643f8e056d8c7e594610ab4a858eee18ef4a2f73a134004',
    'v2stdgs_random_s42':
        'e3c33b918007aeca5643f8e056d8c7e594610ab4a858eee18ef4a2f73a134004',
    'v2stdgs_twin_s42':
        'e3c33b918007aeca5643f8e056d8c7e594610ab4a858eee18ef4a2f73a134004',
}

V2_CONFIG = {
    'v1chk_std_thermal_random_s42':
        '6b78c43202c99c18187bece9c2a4933f4efc9bc5ca4c2df87b0f7136966b64d8',
    'v2abl_mlp_s42':
        '511e78c050d8c2d66fe7b27126626bb306793a13d29b7fff189a679343e66ff2',
    'v2abl_neutralbias_s42':
        '56dbbe9c061d7efddea3c34848fdce6fa8f2b75224710e18cee07a30843b8cce',
    'v2abl_noembed_s42':
        'caa201af7742d41a27ffbce01c8ffd0c482b93b5c22946af9538a0be4a9d7e1f',
    'v2abl_noreinject_s42':
        '63e13f89f5cb34d4d1f7b5b7d535b56f1488b351e305dcd38e6a56e7697ae1f3',
    'v2abl_noscatter_s42':
        '7e162c44a932c4178484565b7de595a2a0cdd8447982b6f342cff2be5dda4b3b',
    'v2abl_rdm_s42':
        'bc2528d4b22b3590cbe6bf23613e05103319651c5eea4735d693740d1be40232',
    'v2d16progorig_const_gs_s42':
        '237cc14394cb2a6495572302fb0f55da04ba59f0a3630f08ed04823113059944',
    'v2d16progorig_vect_gs_s42':
        '0a352668912c53f5e5416c6eb4d632f527bf579aa7d4b8611791d098924fa537',
    'v2d16progstd_const_gs_s42':
        'f5dba290ef31ffc07315fdeb0ac654218b15c549749ae49038d68f3e626a0586',
    'v2d16progstd_vect_gs_s42':
        '6c22ce68cb5fcb2c9f4c8830afad28bc53f2338ee010986129a742644e89d559',
    'v2gram_thermal_random_50_s42':
        'a935d009f628b44d9304254854e2297f6acb50c9e79f605066ec11db7593ec70',
    'v2gs_const_s42':
        '338e24083a65d537f84b44971e7a6458ee9bf7e08494f420fc852467a00a4a64',
    'v2lc_thermal_1e6_s42':
        'f5a55854afd06b8308ebc58b091058db7ffafd34d68cf731170f11f5a8566cbb',
    'v2lc_thermal_2e6_s42':
        '105f99b9f7ea644d47acf3791355625007725b425e34d317bd42e7a10c0fe511',
    'v2lc_thermal_2p5e5_s42':
        'cfdc06791026129557d0d040f32a8e2d80048cb7f264c0db808cc41ce41e066c',
    'v2lc_thermal_5e5_s42':
        'e6ae2ccfb7ff0e6ab5b0e4c504ed5f812c5721c56fad0da0624099a362d0263d',
    'v2mlp_cap_s42':
        'aa377b46f35b3a6cc76d31df2404ac41e88e68d479e6118402063edc903119f3',
    'v2mlp_gs_random_s42':
        'b22fd4e08683a12ef8dfc3208aa3cb02b47d63430709347239c09abc3a2b4f20',
    'v2ogn_mse_s42':
        '995a01cd81d9a6c68f130a241a8790ddaed3b0072fd9db16420f5f23a3a01d17',
    'v2ogn_noE_s42':
        'c5339540533a5122d5f4c5e8db35a050491c49f3388f53fec6c73c672349e788',
    'v2std_thermal_random_s42':
        '0f5d3c8815b0df4a434dbec7acd6cffa593baa3ecd9fcc51c16d055f3993b1fa',
    'v2std_thermal_random_s43':
        'a490f16b6fd68e46e722670d4033134ce5e1ee82eae44948bd5a82619b639a93',
    'v2std_thermal_random_s44':
        'f4aca5dba9e8bb391773e7be763c1ba0671647b6a2503a95c6c22b1151ea060c',
    'v2stdgs_gram_s42':
        '45b32cf604c1b72032a5c915d8f85ad443f1b03d2f1464bfb2eba107e0a24e0c',
    'v2stdgs_random_s42':
        'd731dffb97c04d39a781e1904578ab01f23f068ad1a91d9e0a120cf5e3d09dae',
    'v2stdgs_twin_s42':
        '5d396313853f752999f5374697c26e4816d833238d0c9cfe4b6d36c6a4590b4b',
}

# ---------------------------------------------------------------- (b)
V2_REGISTRY_SNAPSHOT = {
    'v1chk_std_thermal_random_s42':
        '9c72bf5d12038b820ac038d9fdc243d6fa0d45f68faf8dfcb6b7047862565735',
    'v2abl_mlp_s42':
        '1ba6645cdb1e362977f03a1fae90255ca4fa750fd7e66d8f1fd3be0f30f698db',
    'v2abl_neutralbias_s42':
        '805b4b7672deb44a1903593706018d2f11592f25e42baed742f4a5d63e1bca32',
    'v2abl_noembed_s42':
        '1ee6b58035dc7a950c347392c378c36f4b0f82af70259345eac0dc9918d60992',
    'v2abl_noreinject_s42':
        '3e529480d32306202af5087007ebf2e7d342d09cdeaa647f155d860447a9d5fb',
    'v2abl_noscatter_s42':
        'a3b66b9625d984c1f28b1203163dd9a30e3e0f475373f5b053f6b40099429055',
    'v2abl_rdm_s42':
        '2cf68c44dcf557765a763b621e2629a9997970b39594ac9474f0955fd887938a',
    'v2d16progorig_const_gs_s42':
        'd22ae7652a7550e77a922744108079d1a8736874cab5eef1b2fac292f70ad2da',
    'v2d16progorig_const_gs_s43':
        'dce55e574e3d2bfa4bad11a18b7e00be9d07c69a060806a7a7f14f6d29e01c73',
    'v2d16progorig_const_gs_s44':
        '0180af12458996a689d863d49ae74a1395fa848efd78e6cc68c096588dbfb917',
    'v2d16progorig_vect_gs_s42':
        'd1d3d27ef49055fd39ee4f650400b50f509e23134e260f95271e4587023e2db9',
    'v2d16progorig_vect_gs_s43':
        'a48b26b153407bd9a3b46b22fcaa063088cd0c714600d51c269070b86fc92716',
    'v2d16progorig_vect_gs_s44':
        '00fd53d303ed033742516395780cb2ece28d3666625335f45648c1914f365b08',
    'v2d16progstd_const_gs_s42':
        '54d00e19a966c7df864c18d53bba984923a14cb7e112a9d4d2d4af543f8d7204',
    'v2d16progstd_const_gs_s43':
        '1a221608b1e593e945782733e88e200d652ea79e703232e28af153211111df1c',
    'v2d16progstd_const_gs_s44':
        '22bb0a4701d5b949f5979350b181f3caf21b33bf586cb8e14b791af9c686c08c',
    'v2d16progstd_vect_gs_s42':
        'fab7a27d67fd4f2b05c16e658e4ffa7eca28b2aedd481238c7fd5bc2dd1378ec',
    'v2d16progstd_vect_gs_s43':
        '2f74964263e60e48d0c63545dcafc64c5cee0d210eae422ef2bb8f5c38733c0b',
    'v2d16progstd_vect_gs_s44':
        'f563d3809256a08e0f65530ca57a42d40f4ddece01e62b1c57e673df7db58917',
    'v2gram_thermal_random_50_s42':
        '16be3ef09470e427b98928856dc7a6f986b0683403ab5df3c28594bda0129b9d',
    'v2gs_const_s42':
        '90bdb57439342c68dcdc75ae25e8f09293932b5bba1fadb060daf024d45d10f3',
    'v2lc_thermal_1e6_s42':
        '9c5d5964f25fa8a2b776151b7fb23f6bd153beb37c46d35c8338cde854f8ef40',
    'v2lc_thermal_2e6_s42':
        '650ea581753534bb1a23c5cb459a5a094e767628c0137da22ffc7e5020532ac9',
    'v2lc_thermal_2p5e5_s42':
        '1b956c1f74e40c8ccd1421d3f5390ee7adcfa40e9ada6e1d4524b999b06e12c3',
    'v2lc_thermal_5e5_s42':
        '0731dea3d167fbc74c8a2affe9280bf944f2764b04ebe012eeb0a4e60e03dace',
    'v2mlp_cap_s42':
        'e347e4a03208da37cd37bcd2ef9e299e2a53f4821389663108782cbbd866be52',
    'v2mlp_gs_random_s42':
        '9fcf5967c5e91e2a8c373a0eca095f000dc65ba406c453d13041d5d3ed2cb791',
    'v2ogn_mse_s42':
        '2d8e62c2fa06e2686659ec83b3e4dead553c69efe8c3e8c53c6c2c2b2a34459d',
    'v2ogn_noE_s42':
        '367f245b1b6e7ec2b853572d3b8c82bb6b28292b193dd4f141d1e8c39323ebca',
    'v2std_thermal_random_s42':
        'f5b7e4cc6d2f0bff6be996a9584310074784ef1405081ef65b6b560acc68c5ce',
    'v2std_thermal_random_s43':
        '64dbdb91bf87a445d80e098f0ef4b92debd13e06f470041ab9994cf4d128f0b0',
    'v2std_thermal_random_s43_xhost':
        '64dbdb91bf87a445d80e098f0ef4b92debd13e06f470041ab9994cf4d128f0b0',
    'v2std_thermal_random_s44':
        '75a4e760b1b0eafe721982353d809f9a45eed27a27e6e9e7a72901a4f080c8c7',
    'v2stdgs_gram_s42':
        '1ab83f4a8ba803e7fff7f9e324bb5415372bf80b3589962f5bc35cfbf10c00c2',
    'v2stdgs_random_s42':
        'f5784b584d6f2aae0a9a3271c2cb21d7f0393a7334c35c70cdb7117784667896',
    'v2stdgs_random_s43':
        '15bb8af4ad960a4abbbaae2dfbaf96ac539486749e721f3709863e708d3c10c7',
    'v2stdgs_random_s44':
        '6f32ee9565d1dfe87d05454b0feafeecf424946f85c27d1ec7a393b404966316',
    'v2stdgs_twin_s42':
        '80323aab2e7936fdbd78f3859cbff6abd444caad12ef9e9a52810e03ba9f2c77',
    'v2stdgs_twin_s43':
        '0870e00e87b704ec6cb4553b4ca18ac940a0376f0ff16dfc28e49e25359a247c',
    'v2stdgs_twin_s44':
        '615475fd6900ab2c1258bbea92168efce3b4800f4c6438742df2020d9d251c84',
}


# ---------------------------------------------------------------- (c)
def _load_dist_files():
    for cand in (os.path.join(_HERE, "dist_files.sha256"),
                 os.path.join(_HERE, "copy_lists", "dist_files.sha256"),
                 os.path.join(_HERE, "copy_lists", "v2", "dist_files.sha256")):
        if os.path.isfile(cand):
            out = {}
            with open(cand, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        sha, rel = line.split("  ", 1)
                        out[rel] = sha
            return out, cand
    return {}, None


DIST_FILE_SHA256, _DIST_FILES_PATH = _load_dist_files()


def bundle_root():
    """The staged dist this module sits in (tools/ inside the bundle), or None."""
    d = _HERE
    for _ in range(4):
        if (os.path.isdir(os.path.join(d, "lib"))
                and os.path.isdir(os.path.join(d, "checkpoints"))):
            return d
        d = os.path.dirname(d)
    return None


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _selfcheck(root=None):
    ok = True
    missing = sorted(set(V2_FINAL_STATE) - set(V2_REGISTRY_SNAPSHOT))
    for lane in missing:
        ok = False
        print("  [a-b] bundled lane %s is not in the registry snapshot" % lane)
    drift = sorted(l for l in V2_FINAL_STATE
                   if l in V2_REGISTRY_SNAPSHOT and V2_FINAL_STATE[l] != V2_REGISTRY_SNAPSHOT[l])
    for lane in drift:
        ok = False
        print("  [a-b] %s final_state sha differs pull-table vs registry snapshot" % lane)
    print("expected_hashes_v2: %d bundled lanes vs %d registry lanes, %s"
          % (len(V2_FINAL_STATE), len(V2_REGISTRY_SNAPSHOT),
             "ALL CONSISTENT" if not (missing or drift) else "DISCREPANCIES ABOVE"))
    print("expected_hashes_v2: dist_files loaded from %s (%d entries)"
          % (_DIST_FILES_PATH, len(DIST_FILE_SHA256)))
    root = root or bundle_root()
    if root and DIST_FILE_SHA256:
        bad = n = 0
        for rel, sha in sorted(DIST_FILE_SHA256.items()):
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                bad += 1
                print("  [c] MISSING %s" % rel)
            elif _sha256(p) != sha:
                bad += 1
                print("  [c] DRIFT   %s" % rel)
            n += 1
        ok = ok and not bad
        print("expected_hashes_v2: re-hashed %d dist files under %s, %d problems" % (n, root, bad))
    if not ok:
        raise SystemExit("expected_hashes_v2 SELF-CHECK FAILED")
    print("expected_hashes_v2: SELF-CHECK PASS")


if __name__ == "__main__":
    _selfcheck()
