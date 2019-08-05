#Experimental Class for Smiles Enumeration, Iterator and SmilesIterator adapted from Keras 1.2.2
from rdkit import Chem
import numpy as np
import math

class SmilesVectorizer(object):
    """SMILES vectorizer and devectorizer, with support for SMILES enumeration (atom order randomization)
    as data augmentation
    
    :parameter charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
    :parameter pad: Length of the vectorization
    :parameter leftpad: Add spaces to the left of the SMILES
    :parameter isomericSmiles: Generate SMILES containing information about stereogenic centers
    :parameter augment: Enumerate the SMILES during transform
    :parameter canonical: use canonical SMILES during transform (overrides enum)
    :parameter binary: Use RDKit binary strings instead of molecule objects
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=5, maxlength=120, leftpad=True, isomericSmiles=True, augment=True, canonical=False, startchar = '^', endchar = '$', unknownchar = '?', binary=False):
        #Special Characters
        self.startchar = startchar
        self.endchar = endchar
        self.unknownchar = unknownchar
        
        
        #Vectorization and SMILES options
        self.binary = binary
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.augment = augment
        self.canonical = canonical
        self._pad = pad
        self._maxlength = maxlength
        
        #The characterset
        self._charset = None
        self.charset = charset
        
        #Calculate the dimensions
        self.setdims()

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        #Ensure start and endchars are in the charset
        for char in [self.startchar, self.endchar, self.unknownchar]:
            if char not in charset:
                charset = charset + char
        #Set the hidden properties        
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        self.setdims()
        
    @property
    def maxlength(self):
        return self._maxlength
    
    @maxlength.setter
    def maxlength(self, maxlength):
        self._maxlength = maxlength
        self.setdims()
        
    @property
    def pad(self):
        return self._pad
    
    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self.setdims()
        
    def setdims(self):
        """Calculates and sets the output dimensions of the vectorized molecules from the current settings"""
        self.dims = (self.maxlength + self.pad, self._charlen)
    
        
    def fit(self, mols, extra_chars=[]):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.maxlength and self.charset
        
        :parameter smiles: Numpy array or Pandas series containing smiles as strings
        :parameter extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
        """
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        charset = set("".join(list(smiles))) #Is there a smarter way when the list of SMILES is HUGE!
        self.charset = "".join(charset.union(set(extra_chars)))
        self.maxlength = max([len(smile) for smile in smiles])
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        mol = Chem.MolFromSmiles(smiles)
        nmol = self.randomize_mol(mol)
        return Chem.MolToSmiles(nmol, canonical=self.canonical, isomericSmiles=self.isomericSmiles)
    
    def randomize_mol(self, mol):
        """Performs a randomization of the atom order of an RDKit molecule"""
        ans = list(range(mol.GetNumAtoms()))
        np.random.shuffle(ans)
        return Chem.RenumberAtoms(mol,ans)

    def transform(self, mols, augment=None, canonical=None):
        """Perform an enumeration (atom order randomization) and vectorization of a Numpy array of RDkit molecules
        
            :parameter mols: The RDKit molecules to transform in a list or array
            :parameter augment: Override the objects .augment setting
            :parameter canonical: Override the objects .canonical setting
            
            :output: Numpy array with the vectorized molecules with shape [batch, maxlength+pad, charset]
        """
        #TODO make it possible to use both SMILES, RDKit mols and RDKit binary strings in input
        one_hot =  np.zeros([len(mols)] + list(self.dims), dtype=np.int8)

        #Possibl override object settings
        if augment is None:
            augment = self.augment
        if canonical is None:    
            canonical = self.canonical

        for i,mol in enumerate(mols):
            
            #Fast convert from RDKit binary
            if self.binary: mol = Chem.Mol(mol)
            
            if augment:
                mol = self.randomize_mol(mol)
            ss = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=self.isomericSmiles)

            #TODO, Improvement make it robust to too long SMILES strings
            #TODO, Improvement make a "jitter", with random offset within the possible frame
            #TODO, Improvement make it report to many "?"'s

            l = len(ss)
            if self.leftpad:
                offset = self.dims[0]-l-1
            else:
                offset = 1

            for j,c in enumerate(ss):
                charidx = self._char_to_int.get(c, self._char_to_int[self.unknownchar])
                one_hot[i,j+offset,charidx] = 1

            #Pad the start
            one_hot[i,offset-1,self._char_to_int[self.startchar]] = 1
            #Pad the end
            one_hot[i,offset+l:,self._char_to_int[self.endchar]] = 1
            #Pad the space in front of start (Could this lead to funky effects during sampling?)
            #one_hot[i,:offset-1,self._char_to_int[self.endchar]] = 1     
                
        return one_hot

      
    def reverse_transform(self, vect, strip=True):
        """ Performs a conversion of a vectorized SMILES to a SMILES strings
        charset must be the same as used for vectorization.
        
        :parameter vect: Numpy array of vectorized SMILES.
        :parameter strip: Strip start and end tokens from the SMILES string
        """
        #TODO make it possible to take a single vectorized molecule, not a list
        
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            if strip:
                smile = smile.strip(self.startchar + self.endchar)
            smiles.append(smile)
        return np.array(smiles)
    
from rdkit import DataStructs
from rdkit.Chem import AllChem


class ChemceptionVectorizer(object):
    """
    Chemception Vectorizer turns RDKit molecules into 2D chemception "images" with embedded
    molecular information in the different layers.
    
    Data augmentation is possible and controlled via the .augment property.
    The RDKit molecules are rotated randomly in the drawing plane before vectorizaton.
    If .flip is set to true they are also flipped in 50% of the cases. Should not be used
    if theres embedded stereo information. Molecular coordinates are also randomly moved
    0 to 0.5 units in the X and Y direction (jitter).
    
    The dimension of X and Y are int(self.embed*2/self.res) and the number of channels is 8

    The channels contains 3 layers with z-scale atom embedding, 1 layer bond information,
    and 4 layers with Hybridization, Gasteigercharge, Valence and Aromatic flag.


    """
    def __init__(self, embed=16.0, resolution = 0.5, augment=True, flip=True, jitter=0.5, rotation=180):
        """
        :parameter embed:  the size of the embedding array. The embedding is specified in coordinate units which is approximate 1 Aangstrom for RDKit
        :parameter res: This is the resolution or the size of the "pixels" in coordinate space
        :parameter augment: Do rotation, jitter and evt. flip on coordinates before embedding
        :parameter jitter: maximum random movement of coords in X and Y dimensions
        :parameter rotation: Angle in degrees that molecule will be rotated in X and Y plane before
               embedding
        :parameter flip: If True the molecule will be randomly flipped around X axis 50% of times
                (Don't use if Stereo information is embedded).

        """
        self.embed = embed
        self.res = resolution
        self.augment=augment
        self.flip = flip
        self.jitter = jitter
        self.rotation = rotation
        self._xdim = self._ydim = int(self.embed*2/self.res) 
        self._channels = 8
        self.dims = (self._xdim, self._ydim, self._channels) #Tensorflow order, channels are last
        self.z_scales = { 1: [0.91380303970722476, 0.73165158255080509, 0.57733314804364055],
                         5: [0.30595181177678765, 1.0, 0.12627398479525687],
                         6: [0.5188990175807231, 0.78421022815543329, 0.0],
                         7: [1.0, 0.69228268136793891, 1.0],
                         8: [0.8483958101521436, 0.385020123655062, 0.22145975047392397],
                         9: [0.96509304017607156, 0.0, 0.063173404499183961],
                         14: [0.0, 0.71343324836845667, 0.44224053276558339],
                         15: [0.41652648238720696, 0.62420516558479067, 0.76653406760644893],
                         16: [0.33207883139905292, 0.43675896202099707, 0.37174865102406474],
                         17: [0.43492872745120104, 0.080933563161888766, 0.25506635487736889],
                         34: [0.25220492897252711, 0.41968099329754543, 0.42430137548104002],
                         35: [0.35428255473513803, 0.029790668511527674, 0.52407440196510457],
                         53: [0.12908295522613494, 0.057208840097424385, 0.77529938134095022]}        
                
    def fit(self, mols, extra_pad = 5):
        """To be done, it could be nice to be able to precalculate the approximate embedding from a dataset"""
        print("TBD")
        
    def preprocess_mols(self, mols):
        """Calculate GasteigerCharges and 2D coordinates for the molecules
        
        :parameter mols: RDKit molecules to be processed in a list or array
        :returns: preprocessed RDKit molecules in a Numpy array
        
        """
        mols_o = []
        for i,mol in enumerate(mols):
            cmol = Chem.Mol(mol.ToBinary())
            cmol.ComputeGasteigerCharges()
            AllChem.Compute2DCoords(cmol)
            mols_o.append(cmol)
        return np.array(mols_o)
    
    
    def _rotate_coords(self, origin, points, angle):
        """
        Rotate coordinates counterclockwise with specified angle and origin.

        :parameter origin: coordinate for rotation center
        :parameter points: numpy array with 2D coordinates
        :parameter angle: The rotation angle in degrees
        :return: Rotated coordinates
        """
        ox, oy = origin

        coords_o = np.zeros((points.shape[0], 2))

        cosa = math.cos(math.radians(angle))
        sina = math.sin(math.radians(angle))

        coords_o[:,0] = ox + cosa * (points[:,0] - ox) - sina * (points[:,1] - oy)
        coords_o[:,1] = oy + sina * (points[:,0] - ox) + cosa * (points[:,1] - oy)
        return coords_o
    
        
    def vectorize_mol(self, mol, augment=None):
        """Vectorizes a single RDKit mol object into a 2D "image" numpy array

            :parameter mol: RDKit mol with precomputed 2D coords and Gasteiger Charges
            :parameter augment: Overrule objects .augment, useful for consistency in predictions
        
        """
        coords = mol.GetConformer(0).GetPositions()

        if augment is None:
            augment = self.augment

        if augment:
            #Rotate + jitter + flip coords.
            rot = np.random.random()*self.rotation
            coords = self._rotate_coords((0,0), coords, rot)
            
            jitter_x = np.random.random()*2*self.jitter - self.jitter
            jitter_y = np.random.random()*2*self.jitter - self.jitter
            coords = coords + np.array([[jitter_x, jitter_y]])
            
            if self.flip:
                flip_choice = np.random.random()
                #print(flip_choice)
                if flip_choice > 0.5:
                    #print("Flip")
                    coords = coords[:,::-1] #Flip around X-axis
        
        vect = np.zeros(self.dims, dtype='float32')
        #Bonds first
        for i,bond in enumerate(mol.GetBonds()):
            #TODO Future: add stereo-info?
            bondorder = bond.GetBondTypeAsDouble()
            bidx = bond.GetBeginAtomIdx()
            eidx = bond.GetEndAtomIdx()
            bcoords = coords[bidx]
            ecoords = coords[eidx]
            frac = np.linspace(0,1,int(1/self.res*2)) #with a res of 0.5 this should be adequate#TODO implement automatic determination/better line drawing algoritm.
            for f in frac:
                c = (f*bcoords + (1-f)*ecoords)
                idx = int(round((c[0] + self.embed)/self.res))
                idy = int(round((c[1]+ self.embed)/self.res))
                vect[ idx , idy ,0] = bondorder

        #Atoms and properties
        for i,atom in enumerate(mol.GetAtoms()):
                idx = int(round((coords[i][0] + self.embed)/self.res))
                idy = int(round((coords[i][1]+ self.embed)/self.res))
                if (idx > vect.shape[0]) or (idy > vect.shape[1]):
                    print("WARNING: atom outside embedding, consider increasing embedding")
                    continue
                else:
                    scales = self.z_scales[ atom.GetAtomicNum() ]
                    vect[ idx , idy, 1] = scales[0]
                    vect[ idx , idy, 2] = scales[1]
                    vect[ idx , idy, 3] = scales[2]
                    hyptype = atom.GetHybridization().real
                    vect[ idx , idy, 4] = hyptype
                    charge = atom.GetProp("_GasteigerCharge")
                    vect[ idx , idy, 5] = charge
                    valence = atom.GetTotalValence()
                    vect[ idx , idy, 6] = valence
                    isarom = atom.GetIsAromatic()
                    vect[ idx , idy, 7] = isarom

        #Remove Nans if present
        if np.sum(np.isnan(vect)) > 0:
            vect[np.isnan(vect)] = 0
            
        return vect
       
    def transform(self, mols, augment=None):
        """Batch vectorization of molecules 
        
        :parameter mols: RDKit mols with precomputed 2D coords and Gasteiger Charges
        :parameter augment: boolean. Overrides objects .augment setting if not None
        :returns: Numpy array with the chemception images. Shape [number_mols, xdim, ydim, channels]
        
        """
        if len(mols.shape) > 1:
            mols = mols.reshape(-1) #TODO: What if Pandas?
            
        mols_array =  np.zeros([len(mols)] + list(self.dims))
        
        for i,mol in enumerate(mols):
            mols_array[i] = self.vectorize_mol(mol, augment = augment)
            
        return mols_array



class MorganDictVectorizer(object):
    def __init__(self, radius=2, augment=None):
        self.radius = radius
        self.augment = augment #Not used
        self.dims = None
        
    def fit(self, mols):
        """Analyses the molecules and creates the key index for the creation of the dense array"""
        keys=set()
        for mol in mols:
            fp = AllChem.GetMorganFingerprint(mol,self.radius)
            keys.update(fp.GetNonzeroElements().keys())
        keys = list(keys)
        keys.sort()
        self.keys= np.array(keys)
        self.dims = len(self.keys)
        
    def transform_mol(self, mol, misses=False, binary=False):
        """ transforms the mol into a dense array using the fitted keys as index
        
            :parameter mol: the RDKit molecule to be transformed
            :parameter misses: wheter to return the number of key misses for the molecule
         """
        assert type(self.keys) is np.ndarray, "keys are not defined or is not an np.array, has the .fit(mols) function been used?"
        #Get fingerprint as a dictionary
        fp = AllChem.GetMorganFingerprint(mol,self.radius)
        fp_d = fp.GetNonzeroElements()
        
        if binary:
            return np.isin(self.keys, list(fp_d.keys()), assume_unique=True)
        
        #Prepare the array, and set the values
        #TODO is there a way to vectorize and speed up this?
        arr = np.zeros((self.dims,))
        _misses = 0
        for key, value in fp_d.items():
            if key in self.keys:
                arr[self.keys == key] = value
            else:
                _misses = _misses + 1
        
        if misses:
            return arr, _misses
        else:
            return arr
    
    def transform(self, mols, misses=False, binary=False):
        """Transforms a list or array of RDKit molecules into a dense array using the key dictionary (see .fit())
        
        :parameter mols: list or array of RDKit molecules
        :parameter misses: Wheter to return the number of key misses for each molecule
        :parameter binary: only binary bits, ignores misses but is faster
        """
        arr = np.zeros((len(mols), self.dims))
        
        if binary:
            for i, mol in enumerate(mols):
                arr[i,:] = self.transform_mol(mol, binary=True)
            return arr
        
        elif misses:
            _misses = np.zeros((len(mols),1))
            for i, mol in enumerate(mols):
                arr[i,:], _misses[i] = self.transform_mol(mol, misses=misses)
            return arr, _misses
        else:
            for i, mol in enumerate(mols):
                arr[i,:] = self.transform_mol(mol, misses=False)
            return arr
            

class HashedVectorizer(object):
    def __init__(self, nBits=2048, augment=None, **kwargs):
        self.nBits = nBits
        self.augment = augment #Not used
        self.dims = (nBits,)
        self.keys = None
        self.kwargs=kwargs
    
    def get_fp(self,mol):
        """Abstract method, must be overriden in subclass"""
        raise NotImplementedError('Abstract class instantiated, subclass, and override get_fp')
        
    def transform_mol(self, mol):
        """ transforms the molecule into a numpy bit array with the morgan bits

            :parameter mol: the RDKit molecule to be transformed
        """
        fp = self.get_fp(mol)
        arr = np.zeros((self.nBits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def transform(self, mols):
        """Transforms a list or array of RDKit molecules into an array with the Morgan bits
      
        :parameter mols: list or array of RDKit molecules
        """
        
        arr = np.zeros((len(mols), self.nBits))
        for i, mol in enumerate(mols):
            arr[i,:] = self.transform_mol(mol)
        return arr


class HashedAPVectorizer(HashedVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_fp(self, mol):
        return AllChem.GetHashedAtomPairFingerprint(mol,nBits=self.nBits, **self.kwargs)


class HashedMorganVectorizer(HashedVectorizer):
    def __init__(self, radius=2, **kwargs):
        self.radius = radius
        super().__init__(**kwargs)

    def get_fp(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol,self.radius,nBits=self.nBits, **self.kwargs)


class HashedTorsionVectorizer(HashedVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_fp(self, mol):
        return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=self.nBits, **self.kwargs)


#RDKit Fingerprints
class HashedRDKVectorizer(HashedVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_fp(self, mol):
        return Chem.rdmolops.RDKFingerprint(mol, fpSize=self.nBits, **self.kwargs)


#MACCS (Not a hashed fingerprint, but with fixed length
#from rdkit.Chem import MACCSkeys
#fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]

#Avalon

#2D pharmacophore fingerprint

    
