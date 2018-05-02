import logging
import lzma
import io
import threading
import sys
import weakref


CHECK_SIZE = {
	lzma.CHECK_CRC32: 4,
	lzma.CHECK_CRC64: 8,
	lzma.CHECK_SHA256: 32,
}
STREAM_HEADER_LENGTH = 12


class XZBytes(object):
	"""Class to add weakref support to bytes."""
	def __init__(self, data):
		self._data = data

	@property
	def data(self):
		return self._data


class XZBlock(object):
	"""
	Read an XZ block metadata and allow reading uncompressed/compressed size/data.
	"""
	_fp = ...  # type: io.BytesIO
	_log = logging.getLogger(__name__)

	def __init__(self, file, offset, block_check_size, file_lock=None):
		if type(file) == str:
			self._fp = open(file, mode='rb')
		else:
			if hasattr(file, "read") and hasattr(file, "seek"):
				self._fp = file
			else:
				raise TypeError("File must either be a file path or an open file object with read/seek methods.")

		assert issubclass(type(self._fp), io.IOBase)

		self._fp_lock = file_lock or threading.Lock()
		self._fp.seek(offset)
		header = self._fp.read(1)

		if int(header[0]) == 0:
			raise IndexError("This block is the index indicator, not a data block.")

		# Header length + CRC32
		header_length = (int(header[0]) * 4) + 4
		# Exclude the already read size byte.
		header += self._fp.read(header_length - 1)

		self._block_check_size = block_check_size
		self._header = header
		self._header_length = header_length
		self._offset = offset
		self._number_of_filters = self.flag & 0x03
		self._has_compressed_size = True if self.flag & 0x40 else False
		self._has_uncompressed_size = True if self.flag & 0x80 else False

		if self._has_compressed_size is False or self._has_uncompressed_size is False:
			raise NotImplementedError("Can not open block without embedded compressed/uncompressed size.")

		self._check = None
		self._decode_offset = 2
		self._compressed_size = None
		self._compressed_data = None
		self._uncompressed_size = None
		self._uncompressed_data = None

	@property
	def block_check(self) -> bytes:
		with self._fp_lock:
			self._fp.seek(self.offset + self.header_length + self.compressed_size_padded)
			return self._fp.read(self._block_check_size)

	@property
	def compressed_data(self) -> XZBytes:
		"""Return a reference of the compressed bytes in the file."""
		if self._compressed_data is None or self._compressed_data() is None:
			with self._fp_lock:
				self._fp.seek(self.offset + self.header_length)
				compressed_data = self._fp.read(self.compressed_size)
			assert len(compressed_data) == self.compressed_size
			compressed_data = XZBytes(compressed_data)
			self._compressed_data = weakref.ref(compressed_data)

		return self._compressed_data()

	@property
	def block_check_size(self) -> int:
		return self._block_check_size

	@property
	def uncompressed_data(self) -> XZBytes:
		"""Return a reference of the uncompressed data."""
		if self._uncompressed_data is None or self._uncompressed_data() is None:
			uncompressed_data = lzma.decompress(
				self.compressed_data.data,
				format=lzma.FORMAT_RAW,
				filters=[
					{"id": lzma.FILTER_LZMA2, "preset": 7},
				],
			)
			assert len(uncompressed_data) == self.uncompressed_size
			uncompressed_data = XZBytes(uncompressed_data)
			self._uncompressed_data = weakref.ref(uncompressed_data)

		return self._uncompressed_data()

	@property
	def header_crc32(self) -> bytes:
		return self._header[-4:]

	@property
	def header_length(self) -> int:
		return self._header_length

	@property
	def flag(self) -> int:
		return self._header[1]

	@property
	def offset(self)-> int:
		return self._offset

	@property
	def end_offset(self) -> int:
		return self.offset + self.header_length + self.compressed_size_padded + self._block_check_size

	def _evaluate_size(self) -> int:
		head_data = self._header
		size = int(head_data[self._decode_offset]) & 0x7F
		i = 0

		while True:
			i += 1
			x = int(head_data[self._decode_offset+i])

			if x == 0x00:
				raise ValueError("NULL byte while reading a size field.")

			size |= ((x & 0x7F) << (i * 7))

			if x & 0x80 == 0x00:
				break

		self._decode_offset += i + 1
		return size

	@property
	def compressed_size(self) -> int:
		"""Size of the compressed data in block excluding padding."""
		if not self._compressed_size:
			self._compressed_size = self._evaluate_size()

		return self._compressed_size

	@property
	def compressed_size_padded(self) -> int:
		"""Size of the compressed data in block including 0x00 pads for 4 byte alignment."""
		if self.compressed_size % 4 == 0:
			return self.compressed_size
		else:
			return self.compressed_size + (4 - self.compressed_size % 4)

	@property
	def uncompressed_size(self) -> int:
		"""Size of the data if file is to be decompressed by LZMA."""
		if not self._uncompressed_size:
			self._uncompressed_size = self.compressed_size
			self._uncompressed_size = self._evaluate_size()

		return self._uncompressed_size


class XZFile(object):
	"""
	Read XZ file metadata and allow to extract all data or data blocks.
	"""
	_fp = ...  # type: io.FileIO
	_log = logging.getLogger(__name__)

	def __init__(self, file):
		"""
		:param file: Open XZ file object or XZ file path.
		"""
		if type(file) == str:
			self._fp = open(file, mode='rb')
		else:
			if hasattr(file, "read") and hasattr(file, "seek") and hasattr(file, "fileno"):
				self._fp = file
			else:
				raise TypeError("File must either be a file path or an open file object with fileno/read/seek methods.")

		assert issubclass(type(self._fp), io.IOBase)

		self._fp_lock = threading.Lock()
		self._fp.seek(0)
		self._header = self._fp.read(STREAM_HEADER_LENGTH)
		self._check_size = CHECK_SIZE[self._header[7] & 0x0F]

		if not self._header.startswith(b"\xFD7zXZ\x00"):
			self._log.error("Invalid XZ header: %s", self._header)

		self._block_index = [XZBlock(self._fp, STREAM_HEADER_LENGTH, self._check_size)]
		self._block_max = None

	def block_count(self) -> int:
		while self._block_max is None:
			try:
				self.get_block(len(self._block_index))
			except IndexError:
				break

		return self._block_max

	@property
	def header(self) -> bytes:
		"""
		Get the 12 bytes binary XZ file header.
		:rtype: bytes
		"""
		return self._header

	def _get_block_at(self, offset) -> XZBlock:
		return XZBlock(self._fp, offset, self._check_size, file_lock=self._fp_lock)

	def get_block(self, n) -> XZBlock:
		"""
		Get the n block starting from 0, from the first XZ stream as an XZBlock object.
		:param n: Block number.
		:return: XZBlock object
		"""
		i = len(self._block_index)
		while i <= n and self._block_max is None:
			next_offset = self._block_index[i - 1].end_offset
			try:
				last_block = self._get_block_at(next_offset)
			except IndexError:
				self._block_max = i - 1
				raise
			else:
				self._block_index.append(last_block)
				i += 1

		return self._block_index[n]


if __name__ == "__main__":
	f = XZFile(sys.argv[1])
	dc = lzma.LZMADecompressor(lzma.FORMAT_AUTO, None, None)
	block0 = f.get_block(0)
	block1 = f.get_block(1)
	block_count = f.block_count()
	block0_raw = block0.block_data
	block0_data = block0.data
	pass
