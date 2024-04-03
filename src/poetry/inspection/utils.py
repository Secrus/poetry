import re

metadata_regex = re.compile(r"^[^/]*\.dist-info/METADATA$")
pkg_info_regex = re.compile(r"^[a-zA-Z0-9.-]*/PKG-INFO$")
egg_regex = re.compile(r"^[a-zA-Z0-9.-]*(/EGG-INFO)?/PKG-INFO$")