use super::mystd::ffi::{OsStr, OsString};
use super::mystd::fs;
use super::mystd::os::twizzler::ffi::{OsStrExt, OsStringExt};
use super::Either;
use super::{gimli, Context, Endian, EndianSlice, Mapping, Stash, Vec};
use alloc::sync::Arc;
use core::convert::{TryFrom, TryInto};
use core::ops::Deref;
use core::slice;
use core::str;
use object::elf::{ELFCOMPRESS_ZLIB, ELF_NOTE_GNU, NT_GNU_BUILD_ID, SHF_COMPRESSED};
use object::read::elf::{CompressionHeader, FileHeader, SectionHeader, SectionTable, Sym};
use object::read::StringTable;
use object::{BigEndian, Bytes, NativeEndian};
use twizzler_runtime_api::ObjID;

#[cfg(target_pointer_width = "32")]
type Elf = object::elf::FileHeader32<NativeEndian>;
#[cfg(target_pointer_width = "64")]
type Elf = object::elf::FileHeader64<NativeEndian>;

pub(super) fn native_libraries() -> Vec<super::Library> {
    let mut ret = Vec::new();
    let runtime = twizzler_runtime_api::get_runtime();
    let mut id = runtime.get_exeid();
    while let Some(lib) = id.and_then(|e| runtime.get_library(e)) {
        let mut segments = Vec::new();
        let mut idx = 0;
        let bias = lib.dl_info.map(|info| info.addr).unwrap_or(0);
        while let Some(seg) = runtime.get_library_segment(&lib, idx) {
            segments.push(super::LibrarySegment {
                stated_virtual_memory_address: seg.start - bias,
                len: seg.len,
            });
            idx += 1;
        }
        let lib = super::Library { name: lib, segments, bias };
        id = lib.name.next_id;
        ret.push(lib);
    }
    return ret;
}

pub struct Mmap {
    ptr: *mut u8,
    handle: twizzler_runtime_api::ObjectHandle,
    len: usize,
}

impl Deref for Mmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Mapping {
    pub fn new(lib: &twizzler_runtime_api::Library) -> Option<Mapping> {
        let runtime = twizzler_runtime_api::get_runtime();
        let mapping = runtime.get_full_mapping(&lib)?;
        let (start, end) = lib.range;
        let map = Mmap {
            ptr: start as *mut u8,
            handle: mapping,
            len: unsafe { end.offset_from(start) }.try_into().unwrap(),
        };
        Mapping::mk_or_other(map, |map, stash| {
            let object = Object::parse(&map)?;
            Context::new(stash, object, None, None).map(Either::B)
        })
    }
}

struct ParsedSym {
    address: u64,
    size: u64,
    name: u32,
}

pub struct Object<'a> {
    /// Zero-sized type representing the native endianness.
    ///
    /// We could use a literal instead, but this helps ensure correctness.
    endian: NativeEndian,
    /// The entire file data.
    data: &'a [u8],
    sections: SectionTable<'a, Elf>,
    strings: StringTable<'a>,
    /// List of pre-parsed and sorted symbols by base address.
    syms: Vec<ParsedSym>,
}

impl<'a> Object<'a> {
    fn parse(data: &'a [u8]) -> Option<Object<'a>> {
        let elf = Elf::parse(data).ok()?;
        let endian = elf.endian().ok()?;
        let sections = elf.sections(endian, data).ok()?;
        let mut syms = sections.symbols(endian, data, object::elf::SHT_SYMTAB).ok()?;
        if syms.is_empty() {
            syms = sections.symbols(endian, data, object::elf::SHT_DYNSYM).ok()?;
        }
        let strings = syms.strings();

        let mut syms = syms
            .iter()
            // Only look at function/object symbols. This mirrors what
            // libbacktrace does and in general we're only symbolicating
            // function addresses in theory. Object symbols correspond
            // to data, and maybe someone's crazy enough to have a
            // function go into static data?
            .filter(|sym| {
                let st_type = sym.st_type();
                st_type == object::elf::STT_FUNC || st_type == object::elf::STT_OBJECT
            })
            // skip anything that's in an undefined section header,
            // since it means it's an imported function and we're only
            // symbolicating with locally defined functions.
            .filter(|sym| sym.st_shndx(endian) != object::elf::SHN_UNDEF)
            .map(|sym| {
                let address = sym.st_value(endian).into();
                let size = sym.st_size(endian).into();
                let name = sym.st_name(endian);
                ParsedSym { address, size, name }
            })
            .collect::<Vec<_>>();
        syms.sort_unstable_by_key(|s| s.address);
        Some(Object { endian, data, sections, strings, syms })
    }

    pub fn section(&self, stash: &'a Stash, name: &str) -> Option<&'a [u8]> {
        if let Some(section) = self.section_header(name) {
            let mut data = Bytes(section.data(self.endian, self.data).ok()?);

            // Check for DWARF-standard (gABI) compression, i.e., as generated
            // by ld's `--compress-debug-sections=zlib-gabi` flag.
            let flags: u64 = section.sh_flags(self.endian).into();
            if (flags & u64::from(SHF_COMPRESSED)) == 0 {
                // Not compressed.
                return Some(data.0);
            }

            let header = data.read::<<Elf as FileHeader>::CompressionHeader>().ok()?;
            if header.ch_type(self.endian) != ELFCOMPRESS_ZLIB {
                // Zlib compression is the only known type.
                return None;
            }
            let size = usize::try_from(header.ch_size(self.endian)).ok()?;
            let buf = stash.allocate(size);
            decompress_zlib(data.0, buf)?;
            return Some(buf);
        }

        // Check for the nonstandard GNU compression format, i.e., as generated
        // by ld's `--compress-debug-sections=zlib-gnu` flag. This means that if
        // we're actually asking for `.debug_info` then we need to look up a
        // section named `.zdebug_info`.
        if !name.starts_with(".debug_") {
            return None;
        }
        let debug_name = name[7..].as_bytes();
        let compressed_section = self
            .sections
            .iter()
            .filter_map(|header| {
                let name = self.sections.section_name(self.endian, header).ok()?;
                if name.starts_with(b".zdebug_") && &name[8..] == debug_name {
                    Some(header)
                } else {
                    None
                }
            })
            .next()?;
        let mut data = Bytes(compressed_section.data(self.endian, self.data).ok()?);
        if data.read_bytes(8).ok()?.0 != b"ZLIB\0\0\0\0" {
            return None;
        }
        let size = usize::try_from(data.read::<object::U32Bytes<_>>().ok()?.get(BigEndian)).ok()?;
        let buf = stash.allocate(size);
        decompress_zlib(data.0, buf)?;
        Some(buf)
    }

    fn section_header(&self, name: &str) -> Option<&<Elf as FileHeader>::SectionHeader> {
        self.sections.section_by_name(self.endian, name.as_bytes()).map(|(_index, section)| section)
    }

    pub fn search_symtab<'b>(&'b self, addr: u64) -> Option<&'b [u8]> {
        // Same sort of binary search as Windows above
        let i = match self.syms.binary_search_by_key(&addr, |sym| sym.address) {
            Ok(i) => i,
            Err(i) => i.checked_sub(1)?,
        };
        let sym = self.syms.get(i)?;
        if sym.address <= addr && addr <= sym.address + sym.size {
            self.strings.get(sym.name).ok()
        } else {
            None
        }
    }

    pub(super) fn search_object_map(&self, _addr: u64) -> Option<(&Context<'_>, u64)> {
        None
    }

    fn build_id(&self) -> Option<&'a [u8]> {
        for section in self.sections.iter() {
            if let Ok(Some(mut notes)) = section.notes(self.endian, self.data) {
                while let Ok(Some(note)) = notes.next() {
                    if note.name() == ELF_NOTE_GNU && note.n_type(self.endian) == NT_GNU_BUILD_ID {
                        return Some(note.desc());
                    }
                }
            }
        }
        None
    }
}

fn decompress_zlib(input: &[u8], output: &mut [u8]) -> Option<()> {
    use miniz_oxide::inflate::core::inflate_flags::{
        TINFL_FLAG_PARSE_ZLIB_HEADER, TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
    };
    use miniz_oxide::inflate::core::{decompress, DecompressorOxide};
    use miniz_oxide::inflate::TINFLStatus;

    let (status, in_read, out_read) = decompress(
        &mut DecompressorOxide::new(),
        input,
        output,
        0,
        TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF | TINFL_FLAG_PARSE_ZLIB_HEADER,
    );
    if status == TINFLStatus::Done && in_read == input.len() && out_read == output.len() {
        Some(())
    } else {
        None
    }
}

pub(super) fn handle_split_dwarf<'data>(
    _package: Option<&gimli::DwarfPackage<EndianSlice<'data, Endian>>>,
    _stash: &'data Stash,
    _load: addr2line::SplitDwarfLoad<EndianSlice<'data, Endian>>,
) -> Option<Arc<gimli::Dwarf<EndianSlice<'data, Endian>>>> {
    // TODO (dbittman): Add support
    None
}
