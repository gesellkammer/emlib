import appdirs
import os
import json
import logging
import sys
import typing as t
import tabulate
import re
import weakref



__all__ = [
    "makeConfig",
    "getConfig",
    "CheckedDict"
]


logger = logging.getLogger("emlib.conftools")


def _checkValidator(validatordict):
    v = {}
    for key, value in validatordict.items():
        if key.endswith('::choices'):
            value = set(value)
        v[key] = value
    return v


def _isfloaty(value):
    return isinstance(value, (int, float)) or hasattr(value, '__float__')


def _openInStandardApp(path):
    """
    Open path with the app defined to handle it by the user
    at the os level (xdg-open in linux, start in win, open in osx)
    """
    import subprocess
    platform = sys.platform
    if platform == 'linux':
        print(path)
        subprocess.call(["xdg-open", path])
    elif platform == "win32":
        os.startfile(path)
    elif platform == "darwin":
        subprocess.call(["open", path])
    else:
        raise RuntimeError(f"platform {platform} not supported")


def _waitForClick():
    from emlib import dialogs
    dialogs.showinfo("Click OK when finished editing")


def _openInEditor(cfg):
    _openInStandardApp(cfg)
    

class CheckedDict(dict):

    def __init__(self, default: dict, validator: dict=None, callback=None, checkHook=None, precallback=None) -> None:
        """
        A dictionary which checks that the keys and values are valid 
        according to a default dict and a validator

        default: a dict will all default values. A config can accept only
            keys which are already present in the default

        validator: a dict containing choices and types for the keys in the
            default. Given a default like: {'keyA': 'foo', 'keyB': 20},
            a validator could be:

            {'keyA::choices': ['foo', 'bar'],
             'keyB::type': float,
             'keyC::range': (0, 1)
            }

        checkHook: a callback of the form (self, key, value) -> errormsg | None
            Will be called before setting the key to value. It should return "falsey"
            (None, "") if the change is allowed, or an error msg otherwise
        precallback: a callback of the form (self, key, oldvalue, newvalue) -> None
            works as a notification system, can't modify the dictionary configuration itself,
            but can be useful to modify/clear caches, etc.
        """
        self.default = default
        self._allowedkeys = default.keys()
        self._validator = _checkValidator(validator) if validator else None
        self._precallback = None
        self._callback = callback
        self._checkHook = checkHook
        
    def __setitem__(self, key:str, value) -> None:
        if key not in self._allowedkeys:
            raise KeyError(f"Unknown key: {key}")
        oldvalue = self.get(key)
        if oldvalue is not None and oldvalue == value:
            return 
        errormsg = self.checkValue(key, value)
        if errormsg:
            raise ValueError(errormsg)
        if self._precallback:
            self._precallback(self,key, oldvalue, value)

        super().__setitem__(key, value)

        if self._callback is not None:
            self._callback(key, value)
        
    def checkDict(self, d:dict) -> str:
        invalidkeys = [key for key in d if key not in self.default]
        if invalidkeys:
            return f"Some keys are not valid: {invalidkeys}"
        for k, v in d.items():
            errormsg = self.checkValue(k, v)
            if errormsg:
                return errormsg
        return ""
        
    def getChoices(self, key:str) -> t.Optional[t.List]:
        """
        Return a seq. of possible values for key `k`
        or None
        """
        if key not in self._allowedkeys:
            raise KeyError(f"{key} is not a valid key")
        if not self._validator:
            logger.debug("getChoices: validator not set")
            return None
        return self._validator.get(key+"::choices", None)

    def checkValue(self, key: str, value) -> t.Optional[str]:
        """
        Check if value is valid for key

        Returns errormsg. If value is of correct type, errormsg is ""

        Example:

        error = config.checkType(key, value)
        if error:
            print(error)
        """
        choices = self.getChoices(key)
        if choices is not None and value not in choices:
            return f"key should be one of {choices}, got {value}"
        t = self.getType(key)
        if t == float:
            if not _isfloaty(value):
                return f"Expected floatlike for key {key}, got {type(value).__name__}"
        elif t == str and not isinstance(value, (bytes, str)):
            return f"Expected str or bytes for key {key}, got {type(value).__name__}"
        elif not isinstance(value, t):
            return f"Expected {t.__name__} for key {key}, got {type(value).__name__}"
        r = self.getRange(key)
        if r and not (r[0] <= value <= r[1]):
            return f"Value should be within range {r}, got {value}"
        if self._checkHook is not None:
            errormsg = self._checkHook(self, key, value)
            if errormsg:
                return errormsg
        return None

    def getRange(self, key:str) -> t.Tuple:
        if key not in self._allowedkeys:
            raise KeyError(f"{key} is not a valid key")
        if not self._validator:
            logger.debug("getChoices: validator not set")
            return None
        return self._validator.get(key+"::range", None)

    def getType(self, key:str) -> type:
        """
        Returns the expected type for key, as a type

        NB: all numbers are reduced to type float, all strings are of type str,
            otherwise the type of the default value, which can be a collection
            like a list or a dict

        See Also: checkValue
        """
        if self._validator is not None:
            definedtype = self._validator.get(key + "::type")
            if definedtype:
                return definedtype
        defaultvalue = self.default.get(key)
        if defaultvalue is None:
            raise KeyError("Key is not present in default config")
        if isinstance(defaultvalue, bool):
            return bool
        if isinstance(defaultvalue, (int, float)):
            return float
        elif isinstance(defaultvalue, (bytes, str)):
            return str
        else:
            return type(defaultvalue)
        
    def reset(self) -> None:
        """ 
        Resets the config to its default (inplace), and saves it.
        
        Example
        ~~~~~~~

        cfg = getconfig("folder:config")
        cfg = cfg.reset()
        """
        self.update(self.default)
        
    def update(self, d:dict) -> None:
        errormsg = self.checkDict(d)
        if errormsg:
            raise ValueError(f"dict is invalid: {errormsg}")
        super().update(d)


    
class ConfigDict(CheckedDict):

    registry:dict = {}
    
    def __init__(self, name: str, default: dict, validator: dict=None, validHook=None) -> None:
        """
        NB: DONT'T CALL THIS CLASS DIRECTLY. Use makeConfig or getConfig

        This is a persistent dictionary used for configuration

        name: a str of the form `folder:config` or simply `config` if this is 
            an isolated configuration. The json data will be saved at
            $USERCONFIGDIR/folder/config.json
            For instance, in Linux for name mydir:myconfig this would be: 
            ~/.config/mydir/myconfig.json

        default: a dict will all default values. A config can accept only
            keys which are already present in the default

        validator: a dict containing choices and types for the keys in the
            default. Given a default like: {'keyA': 'foo', 'keyB': 20},
            a validator could be:

            {'keyA::choices': ['foo', 'bar'],
             'keyB::type': float
            }

        validHook: if given, if will be called prior to each midification. If have the form
            (self, key, value) -> errormsg, where key and value are the changes which are about to take place
            It should return None if the change is allowed, an errormsg otherwise.
        """
        if not _isValidName(name):
            raise ValueError(f"name {name} is invalid for a config")
        if name in ConfigDict.registry:
            logger.warning("A ConfigDict with the given name already exists!")
        cfg = getConfig(name)
        if cfg and default != cfg.default:
            if not force:
                raise KeyError(f"A config with name {name} was already created "
                               "but the defaults differ")
            else:
                logger.debug(f"makeConfig: config with name {name} already created"
                             "with different defaults. It will be overwritten")
        super().__init__(default=default, validator=validator, callback=self._mycallback, checkHook=validHook)
        self.name = name
        self._allowedkeys = default.keys()
        base, configname = _parseName(name)
        self._base = base
        self._configfile = configname + ".json"
        self._configpath = None
        self._callbackreg = []
        self._ensureWritable()
        self.readConfig(update=True)
        ConfigDict.registry[name] = weakref.ref(self)

    def _mycallback(self, key, value):
        for pattern, func in self._callbackreg:
            if re.match(pattern, key):
                func(self, key, value)
        self._save()

    def register_callback(self, func, pattern=None):
        """
        Register a callback to be fired when a key matching the given pattern is
        changed. If no pattern is given, your function will be called for
        every key.

        func: a function of the form (dict, key, value) -> None

            dict: this ConfigDict itself
            key: the key which was just changed
            value: the new value
        """
        self._callbackreg.append((pattern or r".*", func))

    def _ensureWritable(self) -> None:
        folder, _ = os.path.split(self.getPath())
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save(self, *args):
        path = self.getPath()
        logger.debug(f"Saving config to {path}")
        logger.debug("Config: %s" % json.dumps(self, indent=True))
        f = open(path, "w")
        json.dump(self, f, indent=True)

    def __repr__(self) -> str:
        header = f"Config: {self.name}\n"
        rows = []
        for k, v in self.items():
            info = []
            choices = self.getChoices(k)
            if choices:
                info.append(str(choices))
            keyrange = self.getRange(k)
            if keyrange:
                info.append(str(keyrange))
            keytype = self.getType(k)
            if keytype:
                info.append(keytype.__name__)
                # info.append(str(keytype))
            rows.append((k, v, " ".join(info)))
        return header + tabulate.tabulate(rows)

    def getPath(self) -> str:
        if self._configpath is not None:
            return self._configpath
        self._configpath = path = getPath(self.name)
        return path

    def update(self, d:dict) -> None:
        errormsg = self.checkDict(d)
        if errormsg:
            logger.error(f"ConfigDict: {errormsg}")
            logger.error(f"Reset the dict to a default by removing the file '{self.getPath()}'")
            raise ValueError("dict is invalid")
        super().update(d)

    def openInEditor(self, app=None):
        self._save()
        if app is None:
            _openInEditor(self.getPath())
        else:
            subprocess.call([app, self.getPath()])
        _waitForClick()
        self.readConfig()

    def readConfig(self, update=True) -> dict:
        """
        Read the saved config, update self. This is used internally but it can be usedful
        if the file is changed externally and no monitoring is activated

        * If no saved config (not present or unreadable)
            * if default was given:
                * use default 
            * otherwise:
                * if saved config is unreadable, raise JSONDecodeError
                * if saved config not present, raise FileNotFoundError
        """
        configpath = self.getPath()
        if not os.path.exists(configpath):
            if self.default is None:
                logger.error("No written config found, but default was not set")
                raise FileNotFoundError(f"{configpath} not found")
            logger.debug("Using default config")
            confdict = self.default
        else:
            logger.debug(f"Reading config from disk: {configpath}")
            try:
                confdict = json.load(open(configpath))
                if self.default is None:
                    logger.debug("Setting read config as default")
                    self.default = confdict.copy()
            except json.JSONDecodeError:
                error = sys.exc_info()[0]
                logger.error(f"Could not read config {configpath}: {error}")
                if self.default is not None:
                    logger.debug("Couldn't read config. Using default as fallback")
                    confdict = self.default
                else:
                    logger.error("Couldn't read config. No default given, we give up")
                    raise

        # merge strategy: 
        # * if a key is shared between default and read dict, read dict has priority
        # * if a key is present only in default, it is added
        def merge(readdict, default):
            out = {}
            sharedkeys = readdict.keys() & default.keys()
            for key in sharedkeys:
                out[key] = readdict[key]
            onlyInDefault = default.keys() - readdict.keys()
            for key in onlyInDefault:
                out[key] = default[key]
            return out

        confdict = merge(confdict, self.default)
        
        if update:
            self.update(confdict)
        return confdict


def _makeName(configname: str, base: str = None) -> str:
    if base is not None:
        return f"{base}:{configname}"
    else:
        return f":{configname}"


def _parseName(name: str) -> t.Tuple[str, t.Optional[str]]:
    """
    Returns (configname, base) (which can be None) 
    """
    if ":" not in name:
        base = None
        configname = name
    else:
        base, configname = name.split(":")
        if not base:
            base = None
    return base, configname


def makeConfig(name: str, default: dict, validator: dict=None, force=False, validHook=None) -> ConfigDict:
    """ 
    Example
    ~~~~~~~

    default = {
        'keyA': 10,
        'keyB': 'foo'
    }

    validator = {
        'keyA::type': (int, float),
        'keyB::choices': ['foo', 'bar'],
    }

    config = makeConfig('myapp:myconfig', default=default, validator=validator)

    name:
        unique id of the configuration, a string of the form [base:]configfile
    
        For example, a configfile "foo" with a base "base" will
        be saved as foo.json under folder "base" (in linux, this is
        "~/.config/base/foo.json"
        
        If base is not given, name should be just "foo". In this case, 
        "foo.json" will be saved at the config path instead (in linux, 
        this is "~/.config", in other platforms that may vary)

    default: dict
        a dictionary holding all possible keys with the default values

    validator: dict (optional)
        A validator can specify either the choices for a given key, 
        or the type(s) that a given key can accept. This is specified 
        by appending "::choices" or "::types" to any key in default

        original key    validator key
        -----------------------------
        mykey           mykey::choices
        mykey           mykey::type

        * choices should be a list of possible values the key can accept
        * type should be the type or types (as passed to isinstance) that
          a value can have

        It is not necessary to specify a validation for each key. If no validation
        is given, the values are checked against the default value

    force: bool
        In the case that a given config already exists with different defaults,
        force needs to be set to True to force the creation of the given
        config, otherwise an exception will be raised
    """
    cfg = getConfig(name)
    if cfg and default != cfg.default:
        if not force:
            raise KeyError(f"A config with name {name} was already created "
                           "but the defaults differ")
        else:
            logger.debug(f"makeConfig: config with name {name} already created"
                         "with different defaults. It will be overwritten")
    return ConfigDict(name=name, default=default, validator=validator, validHook=validHook)


def _isValidName(name):
    return re.fullmatch(r"[a-zA-Z0-9\.\:_]+", name)


def _checkName(name):
    if not _isValidName(name):
        raise ValueError(f"{name} is not a valid name for a config: {msg}."
                         " It should contain letters, numbers and any of '.', '_', ':'")

def getConfig(name: str) -> t.Optional[ConfigDict]:
    """ 
    name is the unique id of the configuration: 

    [base]:configfile

    For example, a configfile "foo" with a base "base" will
    be saved as foo.json under folder "base" (in linux, this is
    "~/.config/base/foo.json"

    If base is not given, name should be ":foo", or just "foo"
    In this case, "foo.json" will be saved at the config path 
    instead (in linux, this is "~/.config", in other platforms 
    that may vary)
    """
    _checkName(name)
    confref = ConfigDict.registry.get(name)
    if confref:
        return confref()
    return None
    

def activeConfigs() -> t.Dict[str, ConfigDict]:
    out = {}
    for name, configref in ConfigDict.registry.items():
        config = configref()
        if config:
            out[name] = config
    return out


def removeConfig(name:str) -> bool:
    """
    Remove the given config, returns True if it was found and removed,
    False otherwise
    """
    configpath = getPath(name)
    if os.path.exists(configpath):
        os.remove(configpath)
        return True
    return False 


def getPath(name:str) -> str:
    userconfigdir = appdirs.user_config_dir()
    base, configname = _parseName(name) 
    configfile = configname + ".json"
    if base is not None:
        configdir = os.path.join(userconfigdir, base)
    else:
        configdir = userconfigdir
    return os.path.join(configdir, configfile)


_example_config = {
    'key': 'value1',
    'key2': 3.1415
}

_example_config_validator = {
    'key::choices': ('value1', 'value2'),
}
